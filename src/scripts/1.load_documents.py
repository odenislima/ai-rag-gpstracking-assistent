import os, re, sys, json
from typing import List, Dict, Any, Iterable
from tqdm.auto import tqdm
from pathlib import Path

def get_root_path():
        """Always use the same, absolute (relative to root) paths

        which makes moving the notebooks around easier.
        """
        
        return Path(os.getcwd())

PROJECT_DIR = Path(get_root_path())
assert PROJECT_DIR.exists(), PROJECT_DIR

if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))    

FALLBACK_DIR = PROJECT_DIR / "src/st3405_data/docs/fallback_pages"
FALLBACK_PAGES_MANUAL = [12, 13, 14, 15, 38]  # optional: image-heavy pages you maintain manually
PREPEND_HEADING_IF_MISSING = True             # only add heading if it's not already at the very top

SECTIONS_TOC = [
    {"section_title": "INTRODUÇÃO", "page": 5},
    {"section_title": "1. ESPECIFICAÇÕES TÉCNICAS", "page": 6},
    {"section_title": "1.1. GERAL", "page": 6},
    {"section_title": "1.2. LTE (4G)", "page": 6},
    {"section_title": "1.3. GNSS", "page": 7},
    {"section_title": "2. PERIFÉRICOS", "page": 8},
    {"section_title": "2.1. INSERINDO O CHIP", "page": 8},
    {"section_title": "3. FUNCIONALIDADES", "page": 10},
    {"section_title": "3.1. ANTIFURTO IGNIÇÃO", "page": 10},
    {"section_title": "3.2. ANTIFURTO PORTA", "page": 10},
    {"section_title": "4. DESCRIÇÃO DAS ENTRADAS E SÁIDAS", "page": 11},
    {"section_title": "4.1. ST8300/ST4305", "page": 12},
    {"section_title": "4.2. ST8300R", "page": 13},
    {"section_title": "5. SINALIZAÇÃO DOS LEDS", "page": 14},
    {"section_title": "5.1. LED VERMELHO - GPS", "page": 14},
    {"section_title": "5.2. LED AZUL - GPRS", "page": 15},
    {"section_title": "6. CONFIGURANDO O ST4305", "page": 16},
    {"section_title": "7. PARÂMETRO DE REDE", "page": 17},
    {"section_title": "8. SERIAL RS232", "page": 20},
    {"section_title": "8.1. COMANDOS PELA SERIAL RS232", "page": 22},
    {"section_title": "9. CONFIGURAÇÃO DE ENVIO", "page": 23},
    {"section_title": "10. PARÂMETRO DE SMS", "page": 25},
    {"section_title": "10.1. ST8300/ST4305", "page": 25},
    {"section_title": "10.2. ST8300R", "page": 25},
    {"section_title": "11. PARÂMETROS DE M. SENSOR", "page": 26},
    {"section_title": "12. PARÂMETROS DE TENSÃO", "page": 28},
    {"section_title": "13. PARÂMETROS DE ENTRADA", "page": 29},
    {"section_title": "14. PARÂMETROS DE SAÍDA", "page": 31},
    {"section_title": "15. PERFIS DE ENVIO", "page": 33},
    {"section_title": "16. CONFIGURAÇÃO MODOS DE OPERAÇÃO", "page": 34},
    {"section_title": "17. CONFIGURAÇÕES ADICIONAIS PARA CERCA", "page": 37},
    {"section_title": "18. CERCA POLIGONAL", "page": 38},
    {"section_title": "19. CERCA ELETRÔNICA CIRCULAR", "page": 40},
    {"section_title": "20. ENVIO DE COMANDOS", "page": 42},
    {"section_title": "20.1. LISTA DE COMANDOS DISPONÍVEIS", "page": 42},
    {"section_title": "21. DIAGNÓSTICO", "page": 45},
    {"section_title": "22. PERFIL DE CONFIGURAÇÃO", "page": 46},
    {"section_title": "23. CONFIGURANDO ALERTAS", "page": 48},
    {"section_title": "24. CONFIGURAÇÃO DE CABEÇALHOS (STT E ALT)", "page": 49},
    {"section_title": "24.1. CABEÇALHO DE POSIÇÃO (STT)", "page": 49},
    {"section_title": "24.2. CABEÇALHO DE ALERTAS (ALT)", "page": 51},
    {"section_title": "25. CONFIGURANDO OS MAPEAMENTOS", "page": 52},
    {"section_title": "26. CONFIGURANDO SENHA", "page": 54},
    {"section_title": "27. CONFIGURANDO FINE TRACKING", "page": 55},
    {"section_title": "28. 1- WIRE CONFIG.", "page": 56},
    {"section_title": "29. ADICIONAR ID DO MOTORISTA", "page": 58},
    {"section_title": "30. REMOVER ID DO MOTORISTA", "page": 59},
    {"section_title": "31. LER ID DO MOTORISTA", "page": 60},
    {"section_title": "32. CALIBRAÇÃO DPA POR COMANDO", "page": 61},
    {"section_title": "33. CALIBRAÇÃO DPA POR IGNIÇÃO", "page": 61}
]

# =========================
# Text formatter
# =========================
RE_SPACE = re.compile(r"[ \t]+")
RE_ENDPUNC_CAP = re.compile(r"([.!?])\s*(?=[A-ZÁÉÍÓÚÂÊÔÃÕ0-9])")
RE_HEADING_BREAK = re.compile(r"(?<!\n)(\b\d+(?:\.\d+)*\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕ])")
RE_SECOND_HEADING_INLINE = re.compile(
    r"(\b\d+(?:\.\d+)*\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕ].*?)\s{2,}(\d+(?:\.\d+)*\.\s+[A-ZÁÉÍÓÚÂÊÔÃÕ])"
)
RE_OBS_ANCHORS = re.compile(
    r"(?<!\n)(\b(?:Observaç(?:ão|ões)|IMPORTANTE!?|Atenção!?|Nota|Exemplo)s?:)", re.IGNORECASE
)

def text_formatter(text: str) -> str:
    if not text:
        return ""
    replacements = {"":"- ","":"- ","•":"- ","‣":"- ","◦":"- ","∙":"- ","·":"- ",
                    "–":"-","—":"-","\u00A0":" "}
    for k,v in replacements.items(): text = text.replace(k,v)
    # remove headers/footers/page numbers
    text = re.sub(r"SUNTECH DO BRASIL.+?(?=\n|$)", "", text, flags=re.IGNORECASE)
    text = re.sub(r"P[áa]g?\.*\s*\d+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"Página\s*\d+", "", text, flags=re.IGNORECASE)
    # join & break lines
    text = re.sub(r"([a-z0-9,;])\n(?=[a-z0-9])", r"\1 ", text)
    text = re.sub(r"([.!?])\n(?=[A-ZÁÉÍÓÚÂÊÔÃÕ])", r"\1\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # headings & parameters
    text = re.sub(r"(\b\d+(?:\.\d+)*\.)\s*\n\s*([A-ZÁÉÍÓÚÂÊÔÃÕ])", r"\1 \2", text)  # "5.2.\nIGNIÇÃO" -> "5.2. IGNIÇÃO"
    text = re.sub(r"(\b\d+)\.\s+(\d+\.)", r"\1.\2", text)                           # "1. 1." -> "1.1."
    text = RE_HEADING_BREAK.sub(r"\n\n\1", text)
    text = RE_SECOND_HEADING_INLINE.sub(r"\1\n\2", text)
    text = re.sub(r"(?<!\n)([A-ZÁÉÍÓÚÂÊÔÃÕ][\w\s/.\-]{2,}?\(\d{3,5}\):)", r"\n\1", text)
    text = re.sub(r"(?<!\n)([A-Za-zÁÉÍÓÚÂÊÔÃÕ0-9][^:\n]{2,}:\s*)", r"\n\1", text)
    text = re.sub(r"\s-\s(?=\w)", "\n- ", text)
    text = RE_OBS_ANCHORS.sub(r"\n\n\1", text)
    # cosmetics
    text = re.sub(r"\s*(km/h|Km/h|KM/H|ºC|°C|º|°)\b", r" \1", text)
    text = re.sub(r"\s*–\s*", " – ", text)
    text = re.sub(r"\s*/\s*", " / ", text)
    text = re.sub(r"(\|.+\|)\s{2,}(?=\|)", r"\1\n", text)
    text = RE_SPACE.sub(" ", text)
    text = RE_ENDPUNC_CAP.sub(r"\1 ", text)
    text = re.sub(r"\s+([.,;!?])", r"\1", text)
    text = re.sub(r"[\x00-\x1F]+", "", text)
    text = re.sub(r"[.\-•~*_]{4,}", "", text)
    text = "\n".join(line.strip() for line in text.splitlines() if line.strip())
    return text.strip()

def rough_sentence_like_splits(s: str) -> list:
    if not s.strip():
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕ0-9])", s.strip())
    out = []
    for p in parts:
        out.extend(re.split(r"\n(?=-\s|\b[A-ZÁÉÍÓÚÂÊÔÃÕ][\w ./\-]{2,}?:)", p))
    return [x.strip() for x in out if x.strip()]

def is_fallback_page(text: str, min_text_chars: int = 100) -> bool:
    return not text or len(text.strip()) < min_text_chars

# =========================
# PDF → page texts (PyMuPDF)
# =========================
def open_and_read_pdf(pdf_path: str,
                      skip_pages: Iterable[int] = (),
                      min_text_chars: int = 100) -> List[dict]:
    import fitz
    os.makedirs(FALLBACK_DIR, exist_ok=True)
    doc = fitz.open(pdf_path)
    pages_and_texts = []
    skip_set = set(skip_pages)
    for page_number, page in tqdm(enumerate(doc, start=1), total=len(doc)):
        if page_number in skip_set:
            continue
        txt = text_formatter(page.get_text("text"))
        fb_file = os.path.join(FALLBACK_DIR, f"page_{page_number}.txt")
        if is_fallback_page(txt, min_text_chars):
            if os.path.exists(fb_file):
                txt = open(fb_file, "r", encoding="utf-8").read()
            else:
                with open(fb_file, "w", encoding="utf-8") as f:
                    f.write(f"# Página {page_number}\n\n[Adicionar conteúdo manualmente]\n")
                continue
        if len(txt.strip()) < min_text_chars:
            continue
        sentence_like = rough_sentence_like_splits(txt)
        pages_and_texts.append({
            "page_number": page_number,
            "page_char_count": len(txt),
            "page_word_count": len(txt.split()),
            "page_sentence_count_raw": len(sentence_like),
            "page_token_est": round(len(txt) / 4),
            "text": txt
        })
    return pages_and_texts

# =========================
# TOC normalization & parent-only selection
# =========================
NUM_PREFIX = re.compile(r"^\s*(\d+(?:\.\d+)*)\s*[.)]?\s+")

def _norm_title_for_key(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"\s*\.\s*", ".", s)
    return s.lower()

def _title_depth(title: str) -> int:
    m = NUM_PREFIX.match(title)
    if not m:
        return 0
    return m.group(1).count(".") + 1

def _major_num(title: str):
    m = NUM_PREFIX.match(title)
    if not m:
        return None
    return m.group(1).split(".")[0]  # '22' from '22.1'

def normalize_toc(toc: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    items = []
    for row in toc:
        title = row.get("section_title")
        if not title:
            for k, v in row.items():
                if k.endswith("section_title"):
                    title = v
                    break
        if not title:
            continue
        page = int(row.get("page", -1))
        if page < 0:
            continue
        items.append({
            "section_title": title.strip(),
            "page": page,
            "norm_key": _norm_title_for_key(title),
            "depth": _title_depth(title),
            "major": _major_num(title)
        })
    items.sort(key=lambda x: x["page"])
    return items

def pick_parents_only(toc_norm: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep:
      - depth == 0 (no numeric prefix)
      - depth == 1 (e.g., '1. ...', '2. ...')
      - depth >= 2 ONLY IF its major has no depth-1 parent in the TOC
        (promote '22.1 ...' if '22. ...' is absent)
    """
    has_major_parent = {it["major"] for it in toc_norm if it["depth"] == 1}
    parents = []
    for it in toc_norm:
        if it["depth"] in (0, 1):
            parents.append(it)
        elif it["depth"] >= 2 and it["major"] not in has_major_parent:
            parents.append(it)  # orphan sublevel becomes anchor
    parents.sort(key=lambda x: x["page"])
    return parents

# =========================
# Heading helpers (robust & mid-line safe)
# =========================
def _compile_heading_regex(title: str) -> re.Pattern:
    """
    Tolerant regex for a heading (e.g., '20. ENVIO DE COMANDOS'):
    - flexible spaces around dots in the numeric prefix
    - optional trailing dot after the last numeric segment
    - flexible spaces around '-' and '/' in the title
    - case-insensitive, can match mid-line
    """
    t = title.strip()
    m = NUM_PREFIX.match(t)
    if m:
        num = m.group(1)  # e.g., '20' or '1.2'
        num_pat = r'(?:' + r'\s*\.\s*'.join(map(re.escape, num.split('.'))) + r')\s*\.?'
        after = t[m.end():].strip()
        if after:
            text_pat = re.escape(after)
            text_pat = re.sub(r'\\\s+', r'\\s+', text_pat)
            text_pat = text_pat.replace(r'\-', r'\s*-\s*').replace(r'\/', r'\s*/\s*')
            pat = rf'(?<!\w)\s*{num_pat}\s+{text_pat}'
        else:
            pat = rf'(?<!\w)\s*{num_pat}'
    else:
        text_pat = re.escape(t)
        text_pat = re.sub(r'\\\s+', r'\\s+', text_pat)
        text_pat = text_pat.replace(r'\-', r'\s*-\s*').replace(r'\/', r'\s*/\s*')
        pat = rf'(?<!\w){text_pat}'
    return re.compile(pat, flags=re.IGNORECASE)

def _compile_major_heading_regex(title: str) -> re.Pattern | None:
    """
    Fallback: only require the MAJOR number (e.g., '22') + any optional .x + a capital letter.
    Matches both '22. ...' and '22.1 ...' when TOC/body disagree.
    """
    m = NUM_PREFIX.match(title)
    if not m:
        return None
    major = m.group(1).split('.')[0]
    pat = rf'(?<!\w)\s*{re.escape(major)}(?:\s*\.\s*\d+)*\s*\.?\s+[A-ZÁÉÍÓÚÃÕ]'
    return re.compile(pat, flags=re.IGNORECASE)

def _find_heading_span(combined: str, heading: str) -> tuple[int, int]:
    """
    Return (start, end) char span of the FIRST match of `heading` anywhere in `combined`.
    Tries full heading first; if not found, falls back to MAJOR-number-based regex.
    """
    rx_full = _compile_heading_regex(heading)
    m = rx_full.search(combined)
    if m:
        return m.span()
    rx_major = _compile_major_heading_regex(heading)
    if rx_major:
        m2 = rx_major.search(combined)
        if m2:
            return m2.span()
    return (-1, -1)

def _trim_to_heading_window(combined: str, cur_heading: str, next_heading: str | None) -> str:
    """
    Trim to: [ first(cur_heading) , first(next_heading) )
    Works mid-line and tolerates TOC/body mismatches via major-number fallback.
    """
    if not combined:
        return combined
    s0, s1 = _find_heading_span(combined, cur_heading)
    start = 0 if s0 < 0 else s0
    end = len(combined)
    if next_heading:
        n0, _ = _find_heading_span(combined[start:], next_heading)
        if n0 >= 0:
            end = start + n0
    return combined[start:end].strip()

# =========================
# Assemble sections (PARENT-ONLY), absorbing subsections
# =========================
def assemble_sections_parent_only(raw_pages: List[Dict[str, Any]],
                                  toc_parent_only: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    - Stitch pages from parent start THROUGH next parent start (inclusive),
      then cut at the next parent's in-text heading (robust regex).
    - Parents come from pick_parents_only(); subsections are absorbed.
    """
    page_map = {p["page_number"]: p["text"] for p in raw_pages}
    pages_available = sorted(page_map.keys())
    if not pages_available:
        return []

    out = []
    parent_starts = [t["page"] for t in toc_parent_only]

    for i, entry in enumerate(toc_parent_only):
        start_page = entry["page"]
        next_parent_page = next((p for p in parent_starts[i+1:] if p > start_page),
                                (pages_available[-1] + 1))

        # INCLUDE next parent start page to ensure we capture tail content that spills onto it
        end_page = min(next_parent_page, pages_available[-1])

        # Stitch pages [start_page .. end_page] inclusive
        texts = [page_map[pg] for pg in pages_available if start_page <= pg <= end_page]
        combined = "\n".join(t for t in texts if t and t.strip()).strip()

        # Cut exactly at first occurrence of the next parent heading (mid-line safe)
        next_heading = toc_parent_only[i+1]["section_title"] if (i + 1 < len(toc_parent_only)) else None
        combined = _trim_to_heading_window(combined, entry["section_title"], next_heading)

        # Ensure heading at top (do NOT duplicate if already present)
        if PREPEND_HEADING_IF_MISSING and entry["section_title"]:
            rx_here = _compile_heading_regex(entry["section_title"])
            top2 = "\n".join([ln for ln in combined.splitlines() if ln.strip()][:2])
            if not rx_here.match(combined) and not rx_here.search(top2):
                combined = f"{entry['section_title'].strip()}\n{combined}".strip()

        words = len(combined.split())
        sents = rough_sentence_like_splits(combined)

        out.append({
            "text": combined,
            "page_number": start_page,
            "page_char_count": len(combined),
            "page_word_count": words,
            "page_sentence_count_raw": len(sents),
            "page_token_est": round(len(combined) / 4),
            "section_title": entry["section_title"],
            "section_key": entry["norm_key"],
            "end_page": end_page
        })
    return out

def save_file(pages_and_texts):
    OUT_JSONL = PROJECT_DIR / "src/st3405_data/st_4305_text.jsonl"
    Path(OUT_JSONL).parent.mkdir(parents=True, exist_ok=True)

    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for rec in pages_and_texts:  # produced earlier by run_pipeline(...)
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    len(pages_and_texts), OUT_JSONL

# =========================
# Pipeline (call this in your notebook)
# =========================
def run_pipeline(pdf_path: str,
                 skip_pages: Iterable[int] = (),
                 min_text_chars: int = 100,
                 sections_toc: List[Dict[str, Any]] = SECTIONS_TOC) -> List[Dict[str, Any]]:
    """
    1) Extract & format pages from the PDF (respect skip_pages).
    2) Normalize TOC; keep PARENT-ONLY anchors (depth 0/1), plus orphaned sublevels where no major parent exists.
    3) For each parent, stitch from start_page through next_parent_start_page (inclusive).
    4) Trim to exact in-text window: [current heading .. next parent heading).
    5) Do NOT duplicate headings at the top.
    """
    raw_pages = open_and_read_pdf(pdf_path, skip_pages=skip_pages, min_text_chars=min_text_chars)
    toc_all = normalize_toc(sections_toc)
    toc_parents = pick_parents_only(toc_all)
    sections = assemble_sections_parent_only(raw_pages, toc_parents)
    save_file(sections)
    return sections

if __name__ == "__main__":     

    pdf_path = PROJECT_DIR / "src/st3405_data/docs/st4305_manual.pdf"

    skip_pages=(1, 2, 3, 4)
    pages_and_texts = run_pipeline(pdf_path=pdf_path,
                                   skip_pages=skip_pages,
                                   min_text_chars=100)

    import pandas as pd

    df = pd.DataFrame(pages_and_texts)
    print(df.head())

    # Get stats
    print(df.describe().round(2))

    