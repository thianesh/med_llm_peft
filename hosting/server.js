import express from "express";
import axios from "axios";
import dotenv from "dotenv";
import expressLayouts from "express-ejs-layouts";
dotenv.config();

const app = express();
app.set("view engine", "ejs");
app.set("views", "./views");
app.use(express.urlencoded({ extended: true }));
app.use(expressLayouts);

// ---- Helpers ----
const now = () => process.hrtime.bigint();
const ms = (startNs, endNs) => Number(endNs - startNs) / 1e6;

async function weaviateSearch(query) {
  const start = now();

  // GraphQL hybrid search (works even if you only have BM25). Adjust if you prefer nearText.
  const gql = `
    query Hybrid($query: String!, $limit: Int!) {
      Get {
        ${process.env.WEAVIATE_CLASS}(
          limit: $limit
          hybrid: { query: $query, properties: ["${process.env.WEAVIATE_PROP}"] }
        ) {
          ${process.env.WEAVIATE_PROP}
          _additional {
            id
            score
            distance
          }
        }
      }
    }
  `;

  const headers = { "Content-Type": "application/json" };
  if (process.env.WEAVIATE_API_KEY) {
    headers["Authorization"] = `Bearer ${process.env.WEAVIATE_API_KEY}`;
  }

  const { data } = await axios.post(
    `${process.env.WEAVIATE_URL}/v1/graphql`,
    {
      query: gql,
      variables: { query, limit: Number(process.env.WEAVIATE_LIMIT || 12) }
    },
    { headers }
  );

  const end = now();

  if (data.errors) {
    throw new Error(JSON.stringify(data.errors));
  }

  const hits = (data?.data?.Get?.[process.env.WEAVIATE_CLASS] ?? []).map((h) => ({
    id: h._additional?.id,
    score: h._additional?.score,
    distance: h._additional?.distance,
    text: h[process.env.WEAVIATE_PROP] || ""
  }));

  return {
    hits,
    metrics: {
      weaviate_search_ms: ms(start, end),
      weaviate_count: hits.length
    }
  };
}

function buildPrompt(userQuery, chunks) {
  const context = chunks
    .map((c, i) => `Chunk #${i + 1} (score=${c.score ?? "n/a"}):\n${c.text}`)
    .join("\n\n---\n\n");

  return `You are an expert assistant. Use the context to answer the user's query. 
If the answer isn't in the context, say you are unsure.

User Query:
${userQuery}

Context Chunks:
${context}

Answer:`;
}

async function ollamaGenerate(prompt) {
  const start = now();
  const { data } = await axios.post(
    `${process.env.OLLAMA_URL}/api/generate`,
    {
      model: process.env.OLLAMA_MODEL,
      prompt,
      // stream: false ensures a single JSON response with metrics
      stream: false
    },
    { headers: { "Content-Type": "application/json" }, timeout: 120000 }
  );
  const end = now();

  // Ollama returns metrics like eval_count, eval_duration, prompt_eval_count, total_duration (ns)
  const m = {
    ollama_ms: ms(start, end),
    ollama_model: data?.model,
    ollama_eval_count: data?.eval_count,
    ollama_prompt_eval_count: data?.prompt_eval_count,
    ollama_eval_duration_ms:
      typeof data?.eval_duration === "number" ? data.eval_duration / 1e6 : undefined,
    ollama_prompt_eval_duration_ms:
      typeof data?.prompt_eval_duration === "number" ? data.prompt_eval_duration / 1e6 : undefined,
    ollama_total_duration_ms:
      typeof data?.total_duration === "number" ? data.total_duration / 1e6 : undefined
  };

  return { output: data?.response ?? "", metrics: m };
}

// ---- Routes ----
app.get("/", (req, res) => {
  res.render("index", { defaultQuery: "", error: null });
});


app.post("/search", async (req, res) => {
  const userQuery = (req.body?.q || "").trim();
  if (!userQuery) {
    return res.render("index", { defaultQuery: "", error: "Please enter a query." });
  }

  const t0 = now();
  try {
    // 1) Weaviate search
    const weaviate = await weaviateSearch(userQuery);

    // 2) Build prompt with top chunks (you can trim text length if needed)
    const topChunks = weaviate.hits.slice(0, 8); // also used for the carousel
    const prompt = buildPrompt(userQuery, topChunks);

    // 3) Ollama
    const ollama = await ollamaGenerate(prompt);

    const t1 = now();

    const metrics = {
      ...weaviate.metrics,
      ...ollama.metrics,
      total_request_ms: ms(t0, t1)
    };

    res.render("results", {
      userQuery,
      chunks: topChunks,
      answer: ollama.output,
      metrics
    });
  } catch (err) {
    console.error(err);
    res.render("index", {
      defaultQuery: userQuery,
      error: `Error: ${err?.message || err}`
    });
  }
});

const port = Number(process.env.PORT || 3000);
app.listen(port, () => {
  console.log(`SSR app running on http://localhost:${port}`);
});
