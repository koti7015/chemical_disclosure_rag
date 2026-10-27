"""Settings. Everything can be overridden with environment variables or a .env file."""
import os
import warnings

from dotenv import load_dotenv

load_dotenv()

# Optional CA certificate bundle, e.g. certifi's cacert.pem. Some conda envs set SSL_CERT_FILE to a file
# that does not exist, and then the embedding model download fails. Setting CA_BUNDLE in .env fixes that
# for this app only, and it wins over whatever the shell already has.
CA_BUNDLE = os.getenv("CA_BUNDLE")
if CA_BUNDLE:
    if os.path.isfile(CA_BUNDLE):
        os.environ["SSL_CERT_FILE"] = CA_BUNDLE
        os.environ["REQUESTS_CA_BUNDLE"] = CA_BUNDLE
    else:
        warnings.warn(f"CA_BUNDLE is set but the file does not exist: {CA_BUNDLE}")

DB_PATH = os.getenv("DB_PATH", "app.db")                    # SQLite file, relative to where you run from
CHROMA_PATH = os.getenv("CHROMA_PATH", "chroma_db")         # vector DB folder
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")

FUZZY_CUTOFF = int(os.getenv("FUZZY_CUTOFF", "80"))        # rapidfuzz score, 0-100
MAX_SQL_ROWS = 500          # rows the SQL agent reads back for one sub-question
MAX_ROWS_TO_LLM = 30        # rows we put in the answer prompt
MAX_RERANK = 2000           # hybrid: most SQL rows we re-rank by meaning
WEAK_SIMILARITY = 0.5       # below this a vector match is flagged as weak
MAX_SUB_QUESTIONS = 3
