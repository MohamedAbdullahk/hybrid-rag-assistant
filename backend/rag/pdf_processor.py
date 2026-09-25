import os
import re
from bisect import bisect_right
from typing import List, Tuple
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

# A PDF page with less than this many characters (on average) is treated as a scanned/image PDF
MIN_AVG_CHARS_PER_PAGE = 100

# Section headings, one per line:
#   Level 1 -> "1. Tech Stack Overview", "3. Data Storage Layout"
#   Level 2 -> "7) Playback & Session Service", "A. User & Auth"
LEVEL1_HEADING = re.compile(r"^\s*\d{1,2}\.\s+[A-Z].{2,80}$", re.MULTILINE)
LEVEL2_HEADING = re.compile(r"^\s*(\d{1,2}\)|[A-Z]\.)\s+[A-Z].{2,80}$", re.MULTILINE)

# Decorative underline lines like "-----" or "=====" (no meaning, only noise)
UNDERLINE = re.compile(r"^\s*[=\-]{3,}\s*$", re.MULTILINE)


class PDFProcessor:
    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 150):
        # Used only when a single section is bigger than chunk_size
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )

    # ------------------------------------------------------------------ #
    # Step 1: Read all pages into ONE text (so sections can cross pages)  #
    # ------------------------------------------------------------------ #
    def _read_pages(self, file_path: str) -> Tuple[str, List[int], int, int]:
        reader = PdfReader(file_path)
        parts: List[str] = []
        page_starts: List[int] = []   # character offset where each page begins
        offset = 0
        total_chars = 0

        for page in reader.pages:
            text = page.extract_text() or ""
            total_chars += len(text.strip())
            page_starts.append(offset)
            parts.append(text)
            offset += len(text) + 1   # +1 for the "\n" used to join pages

        return "\n".join(parts), page_starts, total_chars, len(reader.pages)

    @staticmethod
    def _page_of(offset: int, page_starts: List[int]) -> int:
        """Converts a character offset in the full text into a 1-based page number."""
        return bisect_right(page_starts, offset)

    # ------------------------------------------------------------------ #
    # Step 2: Split the full text at every heading line                   #
    # ------------------------------------------------------------------ #
    def _split_sections(self, full_text: str) -> List[Tuple[str, str, int]]:
        """Returns a list of (section_path, section_body, start_offset)."""
        boundaries = []
        for m in LEVEL1_HEADING.finditer(full_text):
            boundaries.append((m.start(), 1, m.group().strip()))
        for m in LEVEL2_HEADING.finditer(full_text):
            boundaries.append((m.start(), 2, m.group().strip()))
        boundaries.sort()

        # No headings found -> treat the whole document as one section
        if not boundaries:
            return [("Document", full_text, 0)]

        sections: List[Tuple[str, str, int]] = []

        # Text before the first heading (e.g. document title)
        if boundaries[0][0] > 0:
            sections.append(("Introduction", full_text[:boundaries[0][0]], 0))

        current_parent = ""
        for i, (start, level, title) in enumerate(boundaries):
            end = boundaries[i + 1][0] if i + 1 < len(boundaries) else len(full_text)
            if level == 1:
                current_parent = title
                path = title
            else:
                path = f"{current_parent} > {title}" if current_parent else title
            sections.append((path, full_text[start:end], start))

        return sections

    # ------------------------------------------------------------------ #
    # Main entry point (same name & return type as before)                #
    # ------------------------------------------------------------------ #
    def extract_and_chunk(self, file_path: str) -> List[Document]:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found at: {file_path}")

        full_text, page_starts, total_chars, num_pages = self._read_pages(file_path)

        # Detect scanned / image-only PDFs
        avg_chars = total_chars / max(num_pages, 1)
        if avg_chars < MIN_AVG_CHARS_PER_PAGE:
            raise ValueError(
                "This PDF looks like a scanned/image PDF (no readable text found). "
                "Please upload a text-based PDF."
            )

        documents: List[Document] = []
        source = os.path.basename(file_path)

        for sec_idx, (path, body, start) in enumerate(self._split_sections(full_text), start=1):
            # Remove "-----" lines and collapse blank lines (avoids tiny heading-only chunks)
            clean_body = re.sub(r"\n\s*\n+", "\n", UNDERLINE.sub("", body)).strip()

            # Skip sections that contain only their heading (e.g. "2. Microservices (High-Level)")
            heading_only = path.split(" > ")[-1]
            if len(clean_body.replace(heading_only, "").strip()) < 20:
                continue

            # Small section -> 1 chunk. Big section -> split further.
            pieces = self.text_splitter.split_text(clean_body)

            chunk_idx = 0
            for piece in pieces:
                # Skip leftover fragments that are only the heading
                if len(piece.replace(heading_only, "").strip()) < 20:
                    continue
                chunk_idx += 1

                # Find the page where this piece starts
                local_pos = body.find(piece[:40])
                page_num = self._page_of(start + max(local_pos, 0), page_starts)

                documents.append(Document(
                    # Section path is added to every chunk so the LLM, BM25 and
                    # embeddings always know which service/section the text belongs to
                    page_content=f"[Section: {path}]\n{piece}",
                    metadata={
                        "page": page_num,
                        "section": path,
                        "chunk_id": f"page_{page_num}_sec_{sec_idx}_chunk_{chunk_idx}",
                        "source": source
                    }
                ))

        return documents