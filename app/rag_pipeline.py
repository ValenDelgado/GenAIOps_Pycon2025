# app/rag_pipeline.py

import os
import re
from langchain.globals import set_verbose, get_verbose

set_verbose(True)  # Si quieres ver logs detallados

from app.llm_provider import get_llm, get_embeddings, get_vectorstore_dir, describe
from app.prompt_registry import DEFAULT_PROMPT, load_prompt_text, list_prompt_options

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain

from dotenv import load_dotenv
import mlflow

load_dotenv()

DATA_DIR = "data/pdfs"
VECTOR_DIR = get_vectorstore_dir()  # vectorstore/openai o vectorstore/fastembed

def load_documents(path=DATA_DIR):
    docs = []
    for file in sorted(os.listdir(path)):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(path, file))
            docs.extend(loader.load())
    return tag_products(docs)


# ---------- Preparación de las fichas técnicas (específico del dominio Ecoline) ----------
# Cada PDF agrupa varias fichas (una por producto). Solo la primera página de cada
# ficha dice "Nombre comercial: X" y todas las páginas repiten el mismo encabezado
# (Código FO-ID-05, fecha de emisión...). Para que la búsqueda funcione:
#   1. se quita el encabezado repetido,
#   2. se parte cada página por secciones (1. DESCRIPCIÓN, 2. COMPOSICIÓN GARANTIZADA...),
#   3. cada fragmento lleva "[Producto: X | Sección: Y]" al inicio.
_NOMBRE_RE = re.compile(r"Nombre\s+comercial\s*:\s*(.+)", re.IGNORECASE)
_SECCION_RE = re.compile(r"^\s*(\d{1,2})\.\s+([A-ZÁÉÍÓÚÑ][A-ZÁÉÍÓÚÑ0-9 ]{3,}?)\s*$", re.MULTILINE)
_ENCABEZADO_RE = re.compile(
    r"^\s*(Documento vigente|FICHA TÉCNICA|Código:.*|Fecha emisi[oó]n.*|Fecha revisi[oó]n.*|"
    r"Versi[oó]n:\s*\d+\s*|P[aá]gina \d+ de \d+)\s*$",
    re.MULTILINE | re.IGNORECASE,
)
PRODUCT_TAGGING = os.getenv("PRODUCT_TAGGING", "1") != "0"


def _limpiar(texto: str) -> str:
    texto = _ENCABEZADO_RE.sub("", texto)
    return re.sub(r"\n\s*\n+", "\n\n", texto).strip()


def tag_products(docs):
    """
    Limpia el encabezado repetido y agrega metadata['product'] a cada página
    según la última ficha vista en ese PDF.
    """
    current, current_source = None, None
    for doc in docs:
        source = doc.metadata.get("source")
        if source != current_source:
            current, current_source = None, source
        match = _NOMBRE_RE.search(doc.page_content)
        if match:
            current = re.sub(r"\s+", " ", match.group(1)).strip()
        doc.metadata["product"] = current or "desconocido"
        if PRODUCT_TAGGING:
            doc.page_content = _limpiar(doc.page_content)
    return docs


def _por_secciones(docs):
    """Parte cada página en secciones; una sección que sigue en la página siguiente conserva su nombre."""
    out, seccion, producto = [], "", None
    for doc in docs:
        if doc.metadata["product"] != producto:
            producto, seccion = doc.metadata["product"], ""
        texto = doc.page_content
        cortes = [(m.start(), f"{m.group(1)}. {m.group(2).strip()}") for m in _SECCION_RE.finditer(texto)]
        tramos = [(0, seccion)] + cortes
        for i, (ini, nombre) in enumerate(tramos):
            fin = tramos[i + 1][0] if i + 1 < len(tramos) else len(texto)
            parte = texto[ini:fin].strip()
            if parte:
                out.append(Document(page_content=parte, metadata={**doc.metadata, "section": nombre}))
            seccion = nombre
    return out


def split_documents(docs, chunk_size, chunk_overlap):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    if not PRODUCT_TAGGING:
        return splitter.split_documents(docs)
    chunks = splitter.split_documents(_por_secciones(docs))
    # descarta fragmentos que son solo el título de una sección (sin contenido)
    chunks = [c for c in chunks if not _SECCION_RE.fullmatch(c.page_content.strip())]
    for chunk in chunks:
        etiqueta = f"Producto: {chunk.metadata.get('product', 'desconocido')}"
        if chunk.metadata.get("section"):
            etiqueta += f" | Sección: {chunk.metadata['section']}"
        chunk.page_content = f"[{etiqueta}]\n{chunk.page_content}"
    return chunks


CHUNK_SIZE = int(os.getenv("CHUNK_SIZE") or 512)
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP") or 50)

def save_vectorstore(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, persist_path=VECTOR_DIR):
    docs = load_documents()
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings()
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    vectordb.save_local(persist_path)

    mlflow.set_experiment("vectorstore_tracking")
    with mlflow.start_run(run_name="vectorstore_build"):
        mlflow.log_param("chunk_size", chunk_size)
        mlflow.log_param("chunk_overlap", chunk_overlap)
        mlflow.log_param("n_chunks", len(chunks))
        mlflow.log_param("n_docs", len(docs))
        mlflow.log_param("product_tagging", PRODUCT_TAGGING)
        mlflow.log_param("n_products", len({d.metadata["product"] for d in docs}))
        mlflow.log_params(describe())
        mlflow.set_tag("vectorstore", persist_path)

def load_vectorstore(chunk_size=512, chunk_overlap=50):
    docs = load_documents()
    chunks = split_documents(docs, chunk_size, chunk_overlap)
    embeddings = get_embeddings()
    return FAISS.from_documents(chunks, embedding=embeddings)

def load_vectorstore_from_disk(persist_path=VECTOR_DIR):
    if not os.path.exists(persist_path):
        raise FileNotFoundError(
            f"No existe el índice en '{persist_path}'. Genéralo primero con: "
            'python -c "from app.rag_pipeline import save_vectorstore; save_vectorstore()"'
        )
    embeddings = get_embeddings()
    return FAISS.load_local(persist_path, embeddings, allow_dangerous_deserialization=True)

def load_prompt(version=DEFAULT_PROMPT):
    """
    Carga el prompt desde el Prompt Registry de MLflow (prompts:/nombre@alias)
    o desde un archivo local de app/prompts/ (nombre sin .txt).
    La URI resuelta queda en prompt.metadata["prompt_uri"].
    """
    prompt_text, uri = load_prompt_text(version)
    return PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_text,
        metadata={"prompt_uri": uri},
    )

def list_prompt_versions():
    """Alias y versiones disponibles en el registro (o archivos locales si no hay registro)."""
    return list_prompt_options()

# Palabras de la pregunta que indican qué sección de la ficha se busca
_SECCIONES_CLAVE = {
    "2.": ["composicion", "garantizada", "contiene", "contenido", "concentracion", "nutriente",
           "nitrogeno", "fosforo", "potasio", "calcio", "magnesio", "azufre", "silicio", "boro",
           "zinc", "carbono", "ph", "densidad", "conductividad", "g/l", "%"],
    "1.": ["registro", "ica", "presentacion", "formulacion", "clase de producto", "toxicologica"],
    "3.": ["aspecto", "color", "estabilidad", "solubilidad", "propiedades"],
    "4.": ["recomendacion", "dosis", "aplicar", "aplica", "mezcla", "compatibilidad", "caneca", "tanque", "bomba"],
    "5.": ["mecanismo", "actua", "funciona", "tecnologia", "neo", "nanofert", "hormona", "fitohormona"],
    "6.": ["beneficio", "sirve", "para que", "ventaja", "suelo", "acidez", "aluminio", "estres", "radicular", "raiz"],
    "7.": ["almacen", "conservar", "guardar"],
    "9.": ["aminoacido", "aminograma", "glicina"],
}


def _sin_tildes(texto: str) -> str:
    return (texto or "").lower().translate(str.maketrans("áéíóúü", "aeiouu"))


def _secciones_pedidas(query: str) -> set[str]:
    q = _sin_tildes(query)
    return {sec for sec, claves in _SECCIONES_CLAVE.items()
            if any(re.search(rf"(?<![a-z]){re.escape(c)}", q) for c in claves)}


# ---------- Preguntas de "qué productos tienen X" (búsqueda por palabra clave) ----------
# Con k fragmentos la búsqueda semántica no puede listar TODOS los productos que
# cumplen algo (p. ej. 6 productos con algas y k=4). Para esas preguntas se buscan
# los términos en todas las fichas y se devuelve un fragmento por producto.
_LISTADO_RE = re.compile(r"(qu[eé]|cu[aá]les?)\s+(son\s+los\s+)?productos|productos\s+(que|con|de\s+ecoline\s+(que|con))|todos\s+los|lista", re.I)
_VACIAS = {"que", "cual", "cuales", "son", "los", "las", "del", "con", "para", "por", "productos", "producto",
           "ecoline", "contienen", "contiene", "tienen", "tiene", "traen", "trae", "llevan", "lleva", "todos",
           "todas", "dame", "lista", "listado", "cuantos", "hay", "una", "uno", "sus", "esta", "estan",
           "portafolio", "incluyen", "incluye", "concentracion", "cuanto", "como", "donde",
           # términos demasiado comunes en las fichas para distinguir productos
           "acido", "acidos", "soluble", "agua", "total", "organico", "organica", "tecnologia"}
_SINONIMOS = {
    "alga": ["alga", "ascophyllum"],
    "giberel": ["giberel", "ag3", "ag₃"],
    "hormona": ["hormona", "citoquinina", "auxina", "giberel", "indolbut"],
    "fitohor": ["hormona", "citoquinina", "auxina", "giberel", "indolbut"],
    "aminoacid": ["aminoacid", "aminograma"],
    "silic": ["silic", "sio2"],
}


def _terminos(query: str) -> list[list[str]]:
    palabras = re.findall(r"[a-z0-9]+", _sin_tildes(query))
    grupos = []
    for w in palabras:
        if len(w) < 4 or w in _VACIAS:
            continue
        raiz = w[:-2] if w.endswith("es") and len(w) > 6 else w.rstrip("s")
        raiz = raiz[:7]
        grupo = next((v for k, v in _SINONIMOS.items() if raiz.startswith(k) or k.startswith(raiz)), [raiz])
        grupos.append(grupo)
    return grupos


def _busqueda_listado(query, resultados, maximo=20):
    """
    Un fragmento por producto que contenga los términos de la pregunta.
    Los términos que aparecen en casi todas las fichas (p. ej. "ácido") se ignoran
    si hay otros más específicos ("giberélico"); entre los específicos basta uno.
    """
    grupos = _terminos(query)
    if not grupos:
        return None
    productos_total = {d.metadata.get("product") for d, _ in resultados}
    textos = [(_sin_tildes(d.page_content), d) for d, _ in resultados]  # ordenados por similitud
    def productos_con(g):
        return {d.metadata.get("product") for t, d in textos if any(x in t for x in g)}
    cobertura = {i: productos_con(g) for i, g in enumerate(grupos)}
    utiles = [i for i in cobertura if cobertura[i]]
    especificos = [i for i in utiles if len(cobertura[i]) <= 0.4 * len(productos_total)]
    usar = [grupos[i] for i in (especificos or utiles)]
    if not usar:
        return None
    por_producto = {}
    for texto, doc in textos:
        prod = doc.metadata.get("product", "desconocido")
        if prod not in por_producto and any(any(x in texto for x in g) for g in usar):
            por_producto[prod] = doc
    return list(por_producto.values())[:maximo]


def _norm(texto: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (texto or "").lower())


class ProductAwareRetriever(BaseRetriever):
    """
    Si la pregunta nombra un producto (p. ej. "composición de BRIXXER"), primero
    devuelve los fragmentos de la ficha de ese producto, ordenados por similitud.
    Si no nombra ninguno, se comporta como la búsqueda semántica normal de FAISS.
    """
    vectordb: object
    k: int = 4

    def _get_relevant_documents(self, query, *, run_manager=None):
        total = len(self.vectordb.index_to_docstore_id)
        resultados = self.vectordb.similarity_search_with_score(query, k=total)
        q = _norm(query)
        productos = {d.metadata.get("product") for d, _ in resultados}
        nombrados = [p for p in productos if p and p != "desconocido" and
                     (_norm(p) in q or (len(_norm(re.sub(r"\d+", "", p))) >= 6 and
                                        _norm(re.sub(r"\d+", "", p)) in q))]
        if not nombrados:
            if _LISTADO_RE.search(query):
                listado = _busqueda_listado(query, resultados)
                if listado and len(listado) > 1:
                    return listado
            return [d for d, _ in resultados[: self.k]]
        # Dentro de cada producto, primero las secciones que pide la pregunta
        # (p. ej. "composición" → sección 2) y luego el resto por similitud.
        pedidas = _secciones_pedidas(query)
        prioridad = lambda d: 0 if any(d.metadata.get("section", "").startswith(s) for s in pedidas) else 1
        # Reparte los k cupos entre los productos nombrados, por turnos
        por_producto = {p: sorted([d for d, _ in resultados if d.metadata.get("product") == p], key=prioridad)
                        for p in nombrados}
        elegidos = []
        while len(elegidos) < self.k and any(por_producto.values()):
            for p in nombrados:
                if por_producto[p] and len(elegidos) < self.k:
                    elegidos.append(por_producto[p].pop(0))
        return elegidos


PRODUCT_BOOST = os.getenv("PRODUCT_BOOST", "1") != "0"


def build_chain(vectordb, prompt_version=DEFAULT_PROMPT, k=4):
    prompt = load_prompt(prompt_version)
    if PRODUCT_BOOST:
        retriever = ProductAwareRetriever(vectordb=vectordb, k=k)
    else:
        retriever = vectordb.as_retriever(search_kwargs={"k": k})
    return ConversationalRetrievalChain.from_llm(
        llm=get_llm(temperature=0),
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt},
        return_source_documents=True  # para mostrar las referencias en la UI
    )
