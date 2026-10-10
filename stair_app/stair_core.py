"""
STAIR (STructure Aware Information Retriever) - Core Implementation
Based on the paper: "STAIR: A novel dataset and LLM based retriever for document structure augmentation"

Architecture:
  Stage 1 - Input Preparation: Extract ToC from PDF/text, map sections to content
  Stage 2 - ToC-Aware Retrieval: Use LLM with full ToC to pick best leaf node
  Stage 3 - Answer Generation: Use retrieved section content + query for RAG
"""

import re
import json
import logging
from dataclasses import dataclass, field, asdict
from typing import Optional
import pymupdf  # PyMuPDF

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ToCNode:
    """Represents one entry in the Table of Contents."""
    id: str                      # unique string id like "1.2.3"
    title: str                   # display title
    level: int                   # depth (0=chapter, 1=section, 2=subsection…)
    page_num: int                # page number in original doc (0-indexed)
    parent_id: Optional[str]     # id of parent node, None for roots
    children_ids: list[str] = field(default_factory=list)
    is_leaf: bool = False        # True if no children
    content: str = ""            # textual content of this section

    def to_dict(self):
        return asdict(self)


@dataclass
class STAIRIndex:
    """Full STAIR index for a document."""
    doc_title: str
    toc_nodes: dict[str, ToCNode]   # id → node
    leaf_ids: list[str]              # ordered list of leaf node ids
    raw_pages: list[str]             # raw text per page

    def get_toc_string(self, max_leaves: int = 200) -> str:
        """
        Render the full ToC as a string for the LLM prompt.
        Leaf nodes are prefixed with 'leaf'.
        Truncates very large ToCs gracefully.
        """
        lines = []
        visited = set()

        def _render(node_id: str, indent: int = 0):
            if node_id in visited:
                return
            visited.add(node_id)
            node = self.toc_nodes.get(node_id)
            if not node:
                return
            prefix = "  " * indent
            leaf_tag = "leaf " if node.is_leaf else ""
            lines.append(f"{prefix}{leaf_tag}{node.id} {node.title}")
            for cid in node.children_ids:
                _render(cid, indent + 1)

        roots = [nid for nid, n in self.toc_nodes.items() if n.parent_id is None]
        for rid in roots:
            _render(rid)

        # If too many leaves, truncate to protect token budget
        leaf_count = sum(1 for l in lines if l.strip().startswith("leaf"))
        if leaf_count > max_leaves:
            # Keep first max_leaves leaves and surrounding context
            kept, leaf_seen = [], 0
            for line in lines:
                if line.strip().startswith("leaf"):
                    leaf_seen += 1
                    if leaf_seen > max_leaves:
                        continue
                kept.append(line)
            lines = kept + [f"  ... (ToC truncated at {max_leaves} leaves)"]

        return "\n".join(lines)

    def get_leaf_titles(self) -> list[str]:
        """Return list of all leaf node titles (used for constrained matching)."""
        return [self.toc_nodes[lid].title for lid in self.leaf_ids if lid in self.toc_nodes]

    def get_leaf_by_title(self, title: str) -> Optional[ToCNode]:
        """Find a leaf node by title (fuzzy match fallback)."""
        title_lower = title.lower().strip()
        # Exact match first
        for lid in self.leaf_ids:
            node = self.toc_nodes.get(lid)
            if node and node.title.lower().strip() == title_lower:
                return node
        # Prefix match
        for lid in self.leaf_ids:
            node = self.toc_nodes.get(lid)
            if node and (node.title.lower().startswith(title_lower[:20]) or
                         title_lower.startswith(node.title.lower()[:20])):
                return node
        # Substring match
        for lid in self.leaf_ids:
            node = self.toc_nodes.get(lid)
            if node and (title_lower in node.title.lower() or
                         node.title.lower() in title_lower):
                return node
        return None

    def get_leaf_by_id_prefix(self, id_str: str) -> Optional[ToCNode]:
        """Find leaf node by its numeric id prefix like '12.9'."""
        id_str = id_str.strip()
        for lid in self.leaf_ids:
            node = self.toc_nodes.get(lid)
            if node and (node.id.startswith(id_str) or lid.startswith(id_str)):
                return node
        return None


# ---------------------------------------------------------------------------
# Stage 1 – Document Parsing & ToC Extraction
# ---------------------------------------------------------------------------

class DocumentParser:
    """
    Parses a PDF or plain-text document and extracts:
      - Table of Contents (hierarchy of sections)
      - Section content mapped to each ToC leaf
    """

    def parse_pdf(self, pdf_path: str) -> STAIRIndex:
        """Main entry point: parse a PDF and return a STAIRIndex."""
        doc = pymupdf.open(pdf_path)
        doc_title = self._infer_title(doc, pdf_path)
        raw_pages = [page.get_text() for page in doc]

        # Try to extract ToC from PDF metadata first
        toc_raw = doc.get_toc()  # list of [level, title, page]
        if toc_raw and len(toc_raw) >= 3:
            logger.info(f"Found PDF ToC with {len(toc_raw)} entries")
            toc_nodes, leaf_ids = self._build_toc_from_metadata(toc_raw, doc, raw_pages)
        else:
            logger.info("No PDF ToC metadata, falling back to text heuristics")
            toc_nodes, leaf_ids = self._build_toc_from_text(raw_pages)

        index = STAIRIndex(
            doc_title=doc_title,
            toc_nodes=toc_nodes,
            leaf_ids=leaf_ids,
            raw_pages=raw_pages,
        )
        logger.info(f"Parsed '{doc_title}': {len(toc_nodes)} nodes, {len(leaf_ids)} leaves")
        return index

    def parse_text(self, text: str, doc_title: str = "Document") -> STAIRIndex:
        """Parse plain text using heuristic heading detection."""
        raw_pages = self._split_into_pages(text)
        toc_nodes, leaf_ids = self._build_toc_from_text(raw_pages)
        return STAIRIndex(
            doc_title=doc_title,
            toc_nodes=toc_nodes,
            leaf_ids=leaf_ids,
            raw_pages=raw_pages,
        )

    # -- private helpers --

    def _infer_title(self, doc, pdf_path: str) -> str:
        meta = doc.metadata or {}
        if meta.get("title"):
            return meta["title"].strip()
        # Fall back to filename
        import os
        return os.path.splitext(os.path.basename(pdf_path))[0]

    def _build_toc_from_metadata(
        self, toc_raw: list, doc, raw_pages: list[str]
    ) -> tuple[dict[str, ToCNode], list[str]]:
        """
        Build ToCNode hierarchy from PyMuPDF ToC metadata.
        toc_raw entries: [level, title, page_1indexed]
        """
        toc_nodes: dict[str, ToCNode] = {}
        counters: dict[int, int] = {}  # level → running count
        parent_stack: list[str] = []   # stack of current ancestor ids

        def _make_id(level: int) -> str:
            counters[level] = counters.get(level, 0) + 1
            # Reset deeper levels
            for l in list(counters.keys()):
                if l > level:
                    del counters[l]
            parts = [str(counters.get(i, 1)) for i in range(1, level + 1)]
            return ".".join(parts)

        nodes_ordered = []
        for entry in toc_raw:
            level, title, page = entry[0], entry[1], entry[2]
            title = title.strip()
            if not title:
                continue

            node_id = _make_id(level)
            # Determine parent
            # Trim parent stack to current level
            while len(parent_stack) >= level:
                parent_stack.pop()
            parent_id = parent_stack[-1] if parent_stack else None

            node = ToCNode(
                id=node_id,
                title=title,
                level=level,
                page_num=max(0, page - 1),
                parent_id=parent_id,
            )
            toc_nodes[node_id] = node
            nodes_ordered.append(node_id)

            if parent_id and parent_id in toc_nodes:
                toc_nodes[parent_id].children_ids.append(node_id)

            parent_stack.append(node_id)

        # Mark leaves (nodes with no children)
        leaf_ids = []
        for nid in nodes_ordered:
            node = toc_nodes[nid]
            if not node.children_ids:
                node.is_leaf = True
                leaf_ids.append(nid)

        # Extract content between consecutive page ranges for each node
        self._assign_content_by_pages(toc_nodes, nodes_ordered, raw_pages)

        return toc_nodes, leaf_ids

    def _assign_content_by_pages(
        self, toc_nodes: dict, ordered_ids: list[str], raw_pages: list[str]
    ) -> None:
        """Assign text content to each section by slicing raw pages."""
        total_pages = len(raw_pages)
        for i, nid in enumerate(ordered_ids):
            node = toc_nodes[nid]
            start_page = node.page_num
            # End page = start of next sibling/next node at same or higher level
            end_page = total_pages
            for j in range(i + 1, len(ordered_ids)):
                next_node = toc_nodes[ordered_ids[j]]
                if next_node.level <= node.level:
                    end_page = next_node.page_num
                    break
            # Collect text
            pages_slice = raw_pages[start_page:max(start_page + 1, end_page)]
            node.content = "\n".join(pages_slice).strip()
            # Clamp very long content
            if len(node.content) > 8000:
                node.content = node.content[:8000] + "\n...[content truncated]"

    def _build_toc_from_text(
        self, raw_pages: list[str]
    ) -> tuple[dict[str, ToCNode], list[str]]:
        """
        Heuristic heading detection from raw text when PDF metadata has no ToC.
        Detects patterns like:
          1. Chapter Title
          1.1 Section
          1.1.1 Subsection
          # Heading (Markdown)
          ALL CAPS headings
        """
        full_text = "\n".join(raw_pages)
        lines = full_text.split("\n")

        heading_re = re.compile(
            r'^(?:'
            r'(?P<numbered>(\d+\.){0,3}\d+)\s+(?P<num_title>[A-Z].{3,80})'  # 1.2.3 Title
            r'|(?P<hash>#{1,4})\s+(?P<hash_title>.{3,80})'                   # ## Markdown
            r'|(?P<caps>^[A-Z][A-Z\s]{10,60}$)'                              # ALL CAPS
            r')'
        )

        entries = []  # (level, title, char_offset)
        char_offset = 0
        for line in lines:
            m = heading_re.match(line.strip())
            if m:
                if m.group("numbered"):
                    num = m.group("numbered")
                    level = num.count(".") + 1
                    title = m.group("num_title").strip()
                    entries.append((level, title, char_offset, num))
                elif m.group("hash"):
                    level = len(m.group("hash"))
                    title = m.group("hash_title").strip()
                    entries.append((level, title, char_offset, ""))
                elif m.group("caps"):
                    title = line.strip().title()
                    entries.append((1, title, char_offset, ""))
            char_offset += len(line) + 1

        if not entries:
            # Fallback: create single root with all content
            root = ToCNode(id="1", title="Document", level=1, page_num=0,
                           parent_id=None, is_leaf=True, content=full_text[:8000])
            return {"1": root}, ["1"]

        # Build hierarchy
        toc_nodes: dict[str, ToCNode] = {}
        ordered_ids: list[str] = []
        counters: dict[int, int] = {}
        parent_stack: list[str] = []

        for i, entry in enumerate(entries):
            level, title = entry[0], entry[1]
            char_off = entry[2]
            orig_num = entry[3]

            counters[level] = counters.get(level, 0) + 1
            for l in list(counters.keys()):
                if l > level:
                    del counters[l]
            node_id = ".".join(str(counters.get(j, 1)) for j in range(1, level + 1))
            if orig_num:
                node_id = orig_num  # Use original numbering if available

            while len(parent_stack) >= level:
                parent_stack.pop()
            parent_id = parent_stack[-1] if parent_stack else None

            # Get content: text until next heading
            end_off = entries[i + 1][2] if i + 1 < len(entries) else len(full_text)
            content = full_text[char_off:end_off].strip()
            if len(content) > 6000:
                content = content[:6000] + "\n...[truncated]"

            node = ToCNode(
                id=node_id, title=title, level=level, page_num=0,
                parent_id=parent_id, content=content
            )
            toc_nodes[node_id] = node
            ordered_ids.append(node_id)
            if parent_id and parent_id in toc_nodes:
                toc_nodes[parent_id].children_ids.append(node_id)
            parent_stack.append(node_id)

        # Mark leaves
        leaf_ids = []
        for nid in ordered_ids:
            node = toc_nodes[nid]
            if not node.children_ids:
                node.is_leaf = True
                leaf_ids.append(nid)

        return toc_nodes, leaf_ids

    def _split_into_pages(self, text: str, chars_per_page: int = 3000) -> list[str]:
        return [text[i:i + chars_per_page] for i in range(0, len(text), chars_per_page)] or [""]


# ---------------------------------------------------------------------------
# Stage 2 – ToC-Aware Retrieval (STAIR Retriever)
# ---------------------------------------------------------------------------

class STAIRRetriever:
    """
    Implements STAIR retrieval:
      Given a query and a STAIRIndex, asks an LLM to select the best leaf node
      from the full ToC. Uses constrained matching to ensure validity.
    """

    RETRIEVAL_PROMPT = """### Document: {doc_title}

You are an expert information retriever. Given the Table of Contents below and a user query, pick the single best leaf section (marked with 'leaf') that most likely contains the answer.

### Table of Contents:
{toc_string}

### Query: {query}

### Instructions:
- You MUST choose ONLY from the leaf sections marked with 'leaf' above.
- Output ONLY the section title, nothing else. No explanation.
- If multiple leaves seem relevant, pick the single most specific one.

### Section:"""

    def __init__(self, ollama_base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:14b"):
        self.ollama_base_url = ollama_base_url
        self.model = model

    def retrieve(
        self,
        query: str,
        index: STAIRIndex,
        top_k: int = 3,
    ) -> list[dict]:
        """
        Retrieve top-k relevant sections for a query.
        Returns list of dicts with node info + content.
        """
        if not index.leaf_ids:
            return []

        toc_str = index.get_toc_string(max_leaves=150)

        # Primary: LLM-based ToC retrieval (STAIR core)
        llm_results = self._llm_retrieve(query, index, toc_str, top_k)

        # Fallback / augmentation: BM25 over leaf content
        bm25_results = self._bm25_retrieve(query, index, top_k)

        # Merge: LLM results first, then BM25 results not already included
        seen_ids = {r["node_id"] for r in llm_results}
        combined = llm_results[:]
        for r in bm25_results:
            if r["node_id"] not in seen_ids and len(combined) < top_k:
                combined.append(r)
                seen_ids.add(r["node_id"])

        return combined[:top_k]

    def _llm_retrieve(
        self, query: str, index: STAIRIndex, toc_str: str, top_k: int
    ) -> list[dict]:
        """Use LLM to select best leaf node from ToC."""
        import requests
        prompt = self.RETRIEVAL_PROMPT.format(
            doc_title=index.doc_title,
            toc_string=toc_str,
            query=query,
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 80,
                "stop": ["\n\n", "###", "Query:", "Section:"],
            }
        }

        results = []
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate",
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            raw_output = resp.json().get("response", "").strip()
            logger.info(f"LLM retrieval raw output: {repr(raw_output)}")

            # Parse response – may be "leaf X.Y Title" or just "Title"
            predicted_title = self._clean_prediction(raw_output)
            node = self._match_leaf(predicted_title, index)
            if node:
                results.append(self._node_to_result(node, "llm", score=1.0))

            # For top_k > 1, also ask for alternatives with modified prompt
            if top_k > 1 and len(results) < top_k:
                alt_results = self._get_alternatives(query, index, toc_str,
                                                     exclude_id=node.id if node else None,
                                                     count=top_k - len(results))
                results.extend(alt_results)

        except Exception as e:
            logger.warning(f"LLM retrieval failed: {e}")

        return results

    def _get_alternatives(
        self, query: str, index: STAIRIndex, toc_str: str,
        exclude_id: Optional[str], count: int
    ) -> list[dict]:
        """Get alternative leaf nodes for top_k > 1."""
        import requests
        alt_prompt = f"""### Document: {index.doc_title}

Given the Table of Contents and query, list {count} MORE relevant leaf sections (different from what was already picked).

### Table of Contents:
{toc_str}

### Query: {query}

### Instructions:
- List exactly {count} leaf section title(s), one per line.
- Only leaf nodes, no explanation.

### Alternative Sections:"""

        payload = {
            "model": self.model,
            "prompt": alt_prompt,
            "stream": False,
            "options": {"temperature": 0.2, "num_predict": 150},
        }
        results = []
        try:
            resp = requests.post(
                f"{self.ollama_base_url}/api/generate", json=payload, timeout=90
            )
            raw = resp.json().get("response", "").strip()
            for line in raw.split("\n"):
                title = self._clean_prediction(line)
                if title:
                    node = self._match_leaf(title, index)
                    if node and node.id != exclude_id:
                        results.append(self._node_to_result(node, "llm_alt", score=0.8))
                        if len(results) >= count:
                            break
        except Exception as e:
            logger.warning(f"Alternative retrieval failed: {e}")
        return results

    def _bm25_retrieve(self, query: str, index: STAIRIndex, top_k: int) -> list[dict]:
        """BM25 retrieval over leaf content as fallback."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return []

        leaf_nodes = [index.toc_nodes[lid] for lid in index.leaf_ids
                      if lid in index.toc_nodes]
        if not leaf_nodes:
            return []

        corpus = [f"{n.title} {n.content}".lower().split() for n in leaf_nodes]
        bm25 = BM25Okapi(corpus)
        scores = bm25.get_scores(query.lower().split())

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                node = leaf_nodes[idx]
                results.append(self._node_to_result(node, "bm25", score=float(scores[idx])))
        return results

    def _clean_prediction(self, raw: str) -> str:
        """Strip 'leaf X.Y ' prefix and clean up LLM output."""
        raw = raw.strip()
        # Remove list markers like "1.", "-", "*"
        raw = re.sub(r'^[\d\.\-\*\•]\s*', '', raw).strip()
        # Remove "leaf X.Y " prefix
        raw = re.sub(r'^leaf\s+[\d\.]+\s*', '', raw, flags=re.IGNORECASE).strip()
        # Remove numeric prefix like "12.9 "
        raw = re.sub(r'^[\d\.]+\s+', '', raw).strip()
        # Take first line if multiple
        raw = raw.split("\n")[0].strip()
        return raw

    def _match_leaf(self, title: str, index: STAIRIndex) -> Optional[ToCNode]:
        """Find best matching leaf node for the predicted title."""
        if not title:
            return None
        node = index.get_leaf_by_title(title)
        if not node:
            node = index.get_leaf_by_id_prefix(title)
        # Try fuzzy: find leaf with most word overlap
        if not node:
            title_words = set(title.lower().split())
            best_score, best_node = 0, None
            for lid in index.leaf_ids:
                n = index.toc_nodes.get(lid)
                if n:
                    n_words = set(n.title.lower().split())
                    score = len(title_words & n_words) / max(len(title_words | n_words), 1)
                    if score > best_score:
                        best_score, best_node = score, n
            if best_score > 0.3:
                node = best_node
        return node

    def _node_to_result(self, node: ToCNode, method: str, score: float) -> dict:
        return {
            "node_id": node.id,
            "title": node.title,
            "level": node.level,
            "content": node.content,
            "page_num": node.page_num + 1,
            "retrieval_method": method,
            "score": score,
        }


# ---------------------------------------------------------------------------
# Stage 3 – RAG Answer Generation
# ---------------------------------------------------------------------------

class STAIRAnswerGenerator:
    """
    Uses retrieved section content to generate a final answer via Ollama.
    This implements the RAG (Retrieval-Augmented Generation) step.
    """

    ANSWER_PROMPT = """You are a helpful assistant answering questions based on document sections.

### Document: {doc_title}
### Retrieved Sections:
{sections_text}

### User Question: {query}

### Instructions:
- Answer based ONLY on the provided sections above.
- Be concise, accurate, and cite the section title when relevant.
- If the sections don't contain enough information, say so clearly.

### Answer:"""

    def __init__(self, ollama_base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:14b"):
        self.ollama_base_url = ollama_base_url
        self.model = model

    def generate(
        self,
        query: str,
        retrieved_sections: list[dict],
        doc_title: str,
        stream: bool = False,
    ):
        """
        Dispatch to streaming or synchronous generation.
        stream=True  → returns a generator of text tokens
        stream=False → returns a full answer string
        """
        if stream:
            return self.generate_stream(query, retrieved_sections, doc_title)
        else:
            return self.generate_sync(query, retrieved_sections, doc_title)

    def generate_sync(
        self,
        query: str,
        retrieved_sections: list[dict],
        doc_title: str,
    ) -> str:
        """Synchronous (non-streaming) answer generation."""
        import requests

        sections_text = self._format_sections(retrieved_sections)
        prompt = self.ANSWER_PROMPT.format(
            doc_title=doc_title,
            sections_text=sections_text,
            query=query,
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 512},
        }
        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip()

    def generate_stream(
        self,
        query: str,
        retrieved_sections: list[dict],
        doc_title: str,
    ):
        """Streaming answer generation – yields text tokens."""
        import requests

        sections_text = self._format_sections(retrieved_sections)
        prompt = self.ANSWER_PROMPT.format(
            doc_title=doc_title,
            sections_text=sections_text,
            query=query,
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "options": {"temperature": 0.3, "num_predict": 512},
        }
        resp = requests.post(
            f"{self.ollama_base_url}/api/generate",
            json=payload,
            stream=True,
            timeout=180,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if line:
                chunk = json.loads(line)
                yield chunk.get("response", "")
                if chunk.get("done"):
                    break

    def _format_sections(self, sections: list[dict]) -> str:
        parts = []
        for i, sec in enumerate(sections, 1):
            parts.append(
                f"[Section {i}: {sec['title']} (p.{sec['page_num']})]:\n{sec['content'][:2000]}"
            )
        return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# High-level STAIR Pipeline
# ---------------------------------------------------------------------------

class STAIRPipeline:
    """
    End-to-end STAIR pipeline: document → index → retrieve → answer
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        retrieval_model: str = "qwen2.5:14b",
        answer_model: str = "qwen2.5:14b",
    ):
        self.parser = DocumentParser()
        self.retriever = STAIRRetriever(ollama_base_url, retrieval_model)
        self.generator = STAIRAnswerGenerator(ollama_base_url, answer_model)
        self._indexes: dict[str, STAIRIndex] = {}  # doc_id → STAIRIndex

    def index_pdf(self, pdf_path: str, doc_id: Optional[str] = None) -> STAIRIndex:
        """Parse and index a PDF. Returns the STAIRIndex."""
        index = self.parser.parse_pdf(pdf_path)
        if doc_id is None:
            import os
            doc_id = os.path.splitext(os.path.basename(pdf_path))[0]
        self._indexes[doc_id] = index
        return index

    def index_text(self, text: str, doc_title: str, doc_id: str) -> STAIRIndex:
        """Parse and index plain text."""
        index = self.parser.parse_text(text, doc_title)
        self._indexes[doc_id] = index
        return index

    def get_index(self, doc_id: str) -> Optional[STAIRIndex]:
        return self._indexes.get(doc_id)

    def list_documents(self) -> list[dict]:
        return [
            {
                "doc_id": doc_id,
                "doc_title": idx.doc_title,
                "num_leaves": len(idx.leaf_ids),
                "num_pages": len(idx.raw_pages),
            }
            for doc_id, idx in self._indexes.items()
        ]

    def query(
        self,
        query: str,
        doc_id: str,
        top_k: int = 3,
        stream_answer: bool = False,
    ) -> dict:
        """
        Full STAIR pipeline: query → retrieve → answer.
        Returns dict with retrieved_sections and answer.
        """
        index = self._indexes.get(doc_id)
        if index is None:
            raise ValueError(f"Document '{doc_id}' not indexed. Call index_pdf() first.")

        # Stage 2: Retrieve
        retrieved = self.retriever.retrieve(query, index, top_k=top_k)

        # Stage 3: Generate
        if stream_answer:
            return {
                "doc_title": index.doc_title,
                "retrieved_sections": retrieved,
                "answer_stream": self.generator.generate(
                    query, retrieved, index.doc_title, stream=True
                ),
            }
        else:
            answer = self.generator.generate(query, retrieved, index.doc_title)
            return {
                "doc_title": index.doc_title,
                "retrieved_sections": retrieved,
                "answer": answer,
            }

    def get_toc(self, doc_id: str) -> Optional[dict]:
        """Return the full ToC tree as a dict for visualization."""
        index = self._indexes.get(doc_id)
        if not index:
            return None
        return {
            "doc_title": index.doc_title,
            "toc_string": index.get_toc_string(),
            "nodes": {nid: n.to_dict() for nid, n in index.toc_nodes.items()},
            "leaf_ids": index.leaf_ids,
        }
