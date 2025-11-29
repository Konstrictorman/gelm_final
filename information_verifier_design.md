# Information Verifier System - Design Document

## Domain Narrowing Recommendation

The model will have the specific focus on NBA domain-information for the following reasons:

### Advantages:

1. **Higher Accuracy**: Domain-specific models perform better on focused knowledge
2. **Structured Data Sources**: Sports have official APIs, databases, and verified statistics
3. **Easier Validation**: Can cross-reference with official records, game logs, player stats
4. **Better Source Quality**: Domain-specific news outlets, official league websites
5. **Reduced Ambiguity**: Less interpretation needed (scores, dates, stats are factual)
6. **Fine-tuning Opportunities**: Can train/ fine-tune models on domain-specific fact-checking datasets

---

## Architecture Overview

```
User Input → Query Processing → Information Retrieval → Evidence Analysis → Classification → Explanation Generation
```

### Components:

1. **LangGraph Workflow** (Orchestration)

   - State management
   - Multi-step verification pipeline
   - Conditional routing based on evidence quality

2. **LangChain** (LLM Integration & Tools)

   - Query enhancement
   - Web search tools
   - Document processing
   - Explanation generation

3. **Hugging Face** (Classification)
   - Fine-tuned fact-checking model
   - Evidence credibility scoring
   - Claim-evidence matching

---

## Detailed Architecture

### State Schema (LangGraph)

```python
class VerificationState(TypedDict):
    user_input: str                    # Original query/claim
    enhanced_query: str                # Enhanced search query
    search_results: List[Document]     # Retrieved documents
    evidence: List[Evidence]           # Extracted evidence
    classification: Literal["real", "fake", "doubtful"]
    confidence: float                  # 0.0 - 1.0
    explanation: str                   # Justification
    sources: List[Source]              # Quoted sources
    metadata: dict                     # Additional info
```

### Workflow Nodes

1. **Query Enhancement Node**

   - Use LLM to expand/refine user query
   - Extract key claims, entities, dates
   - Generate multiple search queries

2. **Information Retrieval Node**

   - Web search (Tavily, Serper, or DuckDuckGo)
   - Domain-specific APIs (e.g., NBA Stats API)
   - Trusted source database lookup

3. **Evidence Extraction Node**

   - Extract relevant facts from search results
   - Cross-reference multiple sources
   - Score evidence quality

4. **Classification Node** (Hugging Face)

   - Compare claim vs evidence
   - Use fine-tuned model for classification
   - Generate confidence score

5. **Explanation Generation Node**

   - LLM generates human-readable explanation
   - Include source citations
   - Highlight conflicting information

6. **Quality Check Node**
   - If confidence < threshold → request more sources
   - If conflicting evidence → mark as "doubtful"
   - Route back to retrieval if needed

---

## Implementation Strategy

### Phase 1: Basic Pipeline

- Simple web search → LLM classification → Explanation

### Phase 2: Multi-Source Verification

- Multiple search queries
- Cross-reference evidence
- Source credibility scoring

### Phase 3: Domain-Specific Enhancement

- Integrate official APIs
- Fine-tune classification model
- Domain-specific source whitelist

### Phase 4: Advanced Features

- Temporal verification (check if info is outdated)
- Claim decomposition (break complex claims)
- Confidence calibration

---

## Technology Stack

### Web Search Tools:

- **Tavily API**: Best for factual queries, returns citations
- **Serper API**: Google search results
- **DuckDuckGo**: Free, privacy-focused
- **LangChain Search Tools**: `TavilySearchResults`, `DuckDuckGoSearchRun`

### Classification Models (Hugging Face):

- **Base**: `facebook/bart-large-mnli` (zero-shot classification)
- **Fine-tuned**: Train on fact-checking datasets (FEVER, PolitiFact)
- **Alternative**: `microsoft/deberta-v3-base` for claim verification

### LLM (LangChain):

- **GPT-4o-mini**: Cost-effective for most tasks
- **GPT-4**: For complex reasoning
- **Claude**: Alternative option

---

## Example Flow (NBA Domain)

**Input**: "LeBron James scored 50 points in the 2024 NBA Finals Game 7"

**Process**:

1. Extract: Player="LeBron James", Event="2024 NBA Finals", Claim="50 points in Game 7"
2. Search: NBA official stats, ESPN, Basketball Reference
3. Evidence: Official box scores, game logs
4. Classification: Compare claim vs official records
5. Output: "FAKE - LeBron James did not score 50 points in Game 7 of 2024 Finals. Official records show..."

---

## Key Design Decisions

1. **Multi-stage verification**: Don't rely on single source
2. **Confidence thresholds**:
   - > 0.8: Real
   - 0.5-0.8: Doubtful
   - <0.5: Fake
3. **Source hierarchy**: Official > Verified news > General web
4. **Explainability**: Always provide reasoning, not just classification
5. **Iterative refinement**: Loop back if evidence insufficient
