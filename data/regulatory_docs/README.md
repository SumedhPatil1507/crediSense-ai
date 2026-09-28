# Regulatory Documents for RAG Copilot

Place source PDFs in this directory before running `python -m src.ingest_regulations`.
The ingest script chunks each PDF, embeds it, and loads it into the ChromaDB vector store
at `data/regulatory_chroma/`.

## Required Documents

| Filename                        | Document                                              | Source |
|---------------------------------|-------------------------------------------------------|--------|
| `rbi_it_framework.pdf`          | RBI Master Direction on IT Framework (2023)           | [RBI](https://www.rbi.org.in/Scripts/BS_ViewMasDirections.aspx) |
| `dpdp_act_2023.pdf`             | Digital Personal Data Protection Act 2023             | [MeitY](https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf) |
| `ecoa_regulation_b.pdf`         | Equal Credit Opportunity Act – Regulation B (CFPB)    | [CFPB](https://www.consumerfinance.gov/rules-policy/regulations/1002/) |
| `fcra_full_text.pdf`            | Fair Credit Reporting Act (FCRA) full text            | [FTC](https://www.ftc.gov/legal-library/browse/statutes/fair-credit-reporting-act) |

## Optional / Additional Documents

| Filename                        | Document                                              |
|---------------------------------|-------------------------------------------------------|
| `rbi_fair_practices_code.pdf`   | RBI Fair Practices Code for NBFCs                     |
| `basel_iii_credit_risk.pdf`     | Basel III – Credit Risk Standardised Approach         |

## How Documents Are Chunked

- Chunk size: **800 tokens** with **100-token overlap**
- Each chunk is tagged with metadata: `source`, `page`, `clause_id` (auto-detected from headings)
- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local, no API key needed)
- Reranker: `cross-encoder/ms-marco-MiniLM-L-6-v2`

## Running the Ingest

```bash
# Full ingest (all PDFs)
python -m src.ingest_regulations

# Single file
python -m src.ingest_regulations --file data/regulatory_docs/rbi_it_framework.pdf

# Force re-ingest (clears existing collection)
python -m src.ingest_regulations --reset
```

## Fallback Mode

If no PDFs are present, the system loads a built-in **seed knowledge base** compiled from
publicly available regulatory summaries. This allows the copilot to function immediately
without requiring PDF downloads.
