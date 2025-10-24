**ARCHITECTURE.md**

---

# ARCHITECTURE LOGICIELLE - SYSTÈME DE MÉMOIRE TEMPORELLE EMPATHIQUE

**Version :** 1.0.0 (Phase 0 - V.E.)  
**Date :** 26 Octobre 2025  
**Auteurs :** Rodin (Claude), Lumi (Gemini), Matthias  
**Statut :** Spécification validée, prête implémentation

---

## TABLE DES MATIÈRES

1. [Vue d'Ensemble](#1-vue-densemble)
2. [Principes Architecturaux](#2-principes-architecturaux)
3. [Architecture Système](#3-architecture-système)
4. [Composants Détaillés](#4-composants-détaillés)
5. [Modèle de Données](#5-modèle-de-données)
6. [Flux de Données](#6-flux-de-données)
7. [Interfaces & APIs](#7-interfaces--apis)
8. [Sécurité & Confidentialité](#8-sécurité--confidentialité)
9. [Performance & Scalabilité](#9-performance--scalabilité)
10. [Déploiement](#10-déploiement)
11. [Monitoring & Observabilité](#11-monitoring--observabilité)
12. [Tests](#12-tests)
13. [Limites & Contraintes](#13-limites--contraintes)
14. [Roadmap Technique](#14-roadmap-technique)

---

## 1. VUE D'ENSEMBLE

### 1.1 Objectif Système

Le système de Mémoire Temporelle Empathique (MTE) transforme un Large Language Model (LLM) conversationnel en un compagnon mémoriel capable de :

- **Capturer exhaustivement** toutes interactions utilisateur-agent-environnement
- **Identifier** des corrélations temporelles et sémantiques récurrentes
- **Enrichir** le contexte conversationnel avec une mémoire structurée
- **Simuler** une continuité relationnelle empathique

**Ce que le système N'EST PAS :**
- ❌ Un oracle causal (ne détermine pas causalité objective)
- ❌ Un système prédictif (Phase 0)
- ❌ Un dispositif médical diagnostique
- ❌ Une IA consciente

**Ce que le système EST :**
- ✅ Une mémoire externalisée structurée
- ✅ Un détecteur de patterns comportementaux
- ✅ Un enrichisseur de contexte conversationnel
- ✅ Un compagnon mémoriel empathique

### 1.2 Cas d'Usage Pilote

**Contexte :** Support sevrage nicotine (Matthias, J1-J90)

**Objectifs Phase 0 :**
1. Valider faisabilité technique capture exhaustive
2. Détecter patterns corrélation basiques
3. Enrichir conversations avec mémoire structurée
4. Valider perception utilisateur de continuité relationnelle

### 1.3 Architecture Haut Niveau

```
┌─────────────────────────────────────────────────────────────┐
│                      UTILISATEUR                            │
│                     (Matthias)                              │
└────────────┬────────────────────────────────────────────────┘
             │
             │ HTTP/WebSocket
             ▼
┌─────────────────────────────────────────────────────────────┐
│                   CLIENT LOGGER                             │
│              (EventLogger Python)                           │
└────────────┬────────────────────────────────────────────────┘
             │
             │ REST API (JSON)
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    LE GREFFIER                              │
│              (Event Ingestion Service)                      │
│    ┌──────────────────────────────────────────┐            │
│    │ - Validation événements                   │            │
│    │ - Analyse sentiment (ML)                  │            │
│    │ - Génération embeddings                   │            │
│    │ - Enrichissement métadonnées              │            │
│    └──────────────────────────────────────────┘            │
└────────────┬────────────────────────────────────────────────┘
             │
             │ Bolt Protocol
             ▼
┌─────────────────────────────────────────────────────────────┐
│                   BASE DE DONNÉES                           │
│                   Neo4j Graph DB                            │
│    ┌──────────────────────────────────────────┐            │
│    │ Nœuds : Events (5000-10000)              │            │
│    │ Arêtes : TEMPOREL_SUITE, CORRELATION     │            │
│    └──────────────────────────────────────────┘            │
└────────────┬────────────────────────────────────────────────┘
             │
             │ Cypher Queries
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    LE JUGE V.E.                             │
│           (Correlation Detection Engine)                    │
│    ┌──────────────────────────────────────────┐            │
│    │ - Co-occurrence temporelle                │            │
│    │ - Similarité sémantique                   │            │
│    │ - Séquençage patterns                     │            │
│    └──────────────────────────────────────────┘            │
│              (Batch Nocturne)                               │
└─────────────────────────────────────────────────────────────┘
             
             ┌──────────────────────────┐
             │   LE MÉMORIALISTE V.E.   │
             │ (Context Enrichment)     │
             └───────────┬──────────────┘
                         │
                         │ Enriched Prompt
                         ▼
             ┌──────────────────────────┐
             │      LLM (Claude)        │
             │   Anthropic API          │
             └───────────┬──────────────┘
                         │
                         │ Response
                         ▼
             ┌──────────────────────────┐
             │     UTILISATEUR          │
             └──────────────────────────┘
```

---

## 2. PRINCIPES ARCHITECTURAUX

### 2.1 Principes Fondamentaux

#### P1 : Séparation des Responsabilités (SoC)

Chaque composant a une responsabilité unique et bien définie :
- **Greffier** : Capture et structuration
- **Juge** : Détection corrélations
- **Mémorialiste** : Enrichissement contexte
- **Base Données** : Persistance et requêtes

#### P2 : Event Sourcing

Tous les événements sont capturés de manière immuable. Le système construit l'état actuel par reconstruction depuis l'historique complet.

**Avantages :**
- Traçabilité totale
- Possibilité replay
- Audit complet
- Pas de perte information

#### P3 : Loose Coupling

Les composants communiquent via interfaces bien définies (REST API, Database queries) permettant :
- Remplacement indépendant composants
- Évolution incrémentale
- Tests isolés

#### P4 : Fail-Safe Design

Le système privilégie la robustesse sur la performance :
- **Perte données = échec critique**
- Redondance (logs + backups)
- Validation stricte inputs
- Dégradation gracieuse

#### P5 : Privacy by Design

Données sensibles (santé mentale, addiction) :
- Anonymisation user_id (hash)
- Chiffrement au repos et en transit
- Contrôles d'accès granulaires
- Conformité RGPD

### 2.2 Principes Techniques

#### T1 : Simplicité Radicale (Phase 0)

**Éviter :**
- Over-engineering prématuré
- Optimisations prématurées
- Abstractions excessives

**Privilégier :**
- Solutions directes
- Code lisible > code clever
- Validations explicites

#### T2 : Observabilité First

Chaque composant produit :
- Logs structurés (JSON)
- Métriques (latence, volumétrie)
- Traces (debugging)

#### T3 : Idempotence

Toutes opérations doivent être rejouables sans effets de bord :
- Création nœud avec UUID
- Upsert arêtes (MERGE Neo4j)
- Requêtes lecture sans side-effects

#### T4 : Configuration Externalisée

Tous paramètres configurables via fichiers/variables environnement :
- URLs services
- Credentials
- Seuils algorithmes
- Fenêtres temporelles

---

## 3. ARCHITECTURE SYSTÈME

### 3.1 Vue Composants

```
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE PRÉSENTATION                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐     ┌──────────────────────┐    │
│  │   CLI Client         │     │   Web Dashboard      │    │
│  │  (Python Script)     │     │   (Future - Phase 1) │    │
│  └──────────────────────┘     └──────────────────────┘    │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ REST API (HTTPS)
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE APPLICATION                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              LE GREFFIER (API Service)               │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │ - Validation & Sanitization                     │ │  │
│  │  │ - Sentiment Analysis (HuggingFace)              │ │  │
│  │  │ - Embedding Generation (SentenceTransformers)   │ │  │
│  │  │ - Event Persistence (Neo4j Driver)              │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │         LE JUGE V.E. (Batch Service)                 │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │ - Temporal Co-occurrence Detector               │ │  │
│  │  │ - Semantic Similarity Matcher                   │ │  │
│  │  │ - Sequential Pattern Finder                     │ │  │
│  │  │ - Correlation Edge Creator                      │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │       LE MÉMORIALISTE V.E. (Query Service)           │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │ - Temporal Context Retriever                    │ │  │
│  │  │ - Semantic Context Retriever                    │ │  │
│  │  │ - Correlation Pattern Retriever                 │ │  │
│  │  │ - Empathic Prompt Builder                       │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       │ Bolt Protocol
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    COUCHE DONNÉES                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Neo4j Graph Database                    │  │
│  │  ┌────────────────────────────────────────────────┐ │  │
│  │  │ Nodes:                                          │ │  │
│  │  │   - Event (Label)                               │ │  │
│  │  │                                                  │ │  │
│  │  │ Edges:                                          │ │  │
│  │  │   - TEMPOREL_SUITE (succession chronologique)  │ │  │
│  │  │   - CORRELATION_OBSERVEE (patterns détectés)   │ │  │
│  │  │                                                  │ │  │
│  │  │ Indexes:                                        │ │  │
│  │  │   - event_id (UNIQUE)                           │ │  │
│  │  │   - event_timestamp                             │ │  │
│  │  │   - event_user_id                               │ │  │
│  │  └────────────────────────────────────────────────┘ │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    COUCHE EXTERNE                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────┐     ┌──────────────────────┐    │
│  │   Anthropic API      │     │  HuggingFace Models  │    │
│  │   (Claude LLM)       │     │  (Sentiment, Embed)  │    │
│  └──────────────────────┘     └──────────────────────┘    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 Vue Déploiement (Phase 0)

```
┌─────────────────────────────────────────────────────────────┐
│                    MACHINE DÉVELOPPEMENT                    │
│                    (Local - Matthias)                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Docker Compose Stack                    │  │
│  │                                                       │  │
│  │  ┌─────────────────┐      ┌─────────────────┐      │  │
│  │  │   Container:    │      │   Container:    │      │  │
│  │  │   Neo4j         │◄─────┤   Greffier API  │      │  │
│  │  │   (Port 7687)   │      │   (Port 8000)   │      │  │
│  │  └─────────────────┘      └─────────────────┘      │  │
│  │                                                       │  │
│  │  ┌─────────────────┐      ┌─────────────────┐      │  │
│  │  │   Container:    │      │   Script:       │      │  │
│  │  │   Juge Batch    │      │   EventLogger   │      │  │
│  │  │   (Cron 3AM)    │      │   (CLI Client)  │      │  │
│  │  └─────────────────┘      └─────────────────┘      │  │
│  │                                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Volumes Persistants                     │  │
│  │  - ./data/neo4j     (Base de données)                │  │
│  │  - ./logs           (Application logs)               │  │
│  │  - ./backups        (Backups quotidiens)             │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## 4. COMPOSANTS DÉTAILLÉS

### 4.1 LE GREFFIER (Event Ingestion Service)

#### 4.1.1 Responsabilités

1. **Validation Événements**
   - Vérification schéma JSON (Pydantic)
   - Sanitization inputs (XSS, injection)
   - Validation types et formats

2. **Enrichissement Automatique**
   - Analyse sentiment (ML model)
   - Génération embeddings sémantiques
   - Extraction entités (NER - optionnel Phase 0)
   - Timestamps précis (ISO 8601)

3. **Persistance**
   - Création nœuds Neo4j
   - Gestion transactions
   - Logging opérations

4. **API REST**
   - Endpoint POST /log_event
   - Authentication (API key Phase 0)
   - Rate limiting (optionnel Phase 0)

#### 4.1.2 Stack Technique

**Framework :** FastAPI 0.104.1

**Justification :**
- Performance élevée (ASGI, async)
- Type hints Python (validation automatique)
- Documentation auto (OpenAPI/Swagger)
- Écosystème mature

**Dépendances Principales :**
```python
fastapi==0.104.1           # Framework web
uvicorn==0.24.0            # ASGI server
pydantic==2.5.0            # Validation données
neo4j==5.14.0              # Driver database
transformers==4.35.0       # ML models (sentiment)
sentence-transformers==2.2.2  # Embeddings
python-dotenv==1.0.0       # Configuration
tenacity==8.2.3            # Retry logic
```

#### 4.1.3 Architecture Interne

```
Greffier/
├── main.py                 # Application FastAPI, endpoints
├── models/
│   ├── __init__.py
│   ├── event.py            # Pydantic models (Event, EventCreate)
│   └── response.py         # Pydantic models (LogResponse)
├── services/
│   ├── __init__.py
│   ├── validation.py       # Validation & sanitization
│   ├── sentiment.py        # Analyse sentiment (HuggingFace)
│   ├── embedding.py        # Génération embeddings
│   └── persistence.py      # Neo4j operations
├── database/
│   ├── __init__.py
│   ├── connection.py       # Neo4j driver singleton
│   └── queries.py          # Cypher queries
├── utils/
│   ├── __init__.py
│   ├── logger.py           # Logging structuré
│   └── config.py           # Configuration management
├── tests/
│   ├── __init__.py
│   ├── test_validation.py
│   ├── test_sentiment.py
│   └── test_persistence.py
├── requirements.txt
├── Dockerfile
└── .env.example
```

#### 4.1.4 Modèles de Données (Pydantic)

```python
# models/event.py
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Dict, List
from datetime import datetime
from uuid import UUID, uuid4

class EventContenu(BaseModel):
    """Contenu d'un événement"""
    texte: str = Field(..., min_length=1, max_length=10000)
    sentiment_detecte: Optional[float] = Field(None, ge=-1.0, le=1.0)
    intensite: Optional[float] = Field(None, ge=0.0, le=1.0)
    tags: List[str] = Field(default_factory=list)
    embeddings: Optional[List[float]] = None
    
    @field_validator('texte')
    @classmethod
    def sanitize_texte(cls, v: str) -> str:
        """Sanitization basique"""
        # Supprimer caractères de contrôle
        return ''.join(char for char in v if ord(char) >= 32 or char in '\n\r\t')

class EventContexte(BaseModel):
    """Contexte d'un événement"""
    domaine: str = Field(..., min_length=1)
    phase: Optional[str] = None
    heure_journee: Optional[int] = Field(None, ge=0, le=23)
    jour_semaine: Optional[int] = Field(None, ge=0, le=6)
    metadata: Dict = Field(default_factory=dict)

class EventCreate(BaseModel):
    """Événement à créer (input API)"""
    agent: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., min_length=1, max_length=255)
    session_id: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1, max_length=100)
    contenu: EventContenu
    contexte: EventContexte

class Event(EventCreate):
    """Événement complet (avec métadonnées système)"""
    id: UUID = Field(default_factory=uuid4)
    timestamp_start: datetime = Field(default_factory=datetime.utcnow)
    timestamp_end: Optional[datetime] = None
    
class LogResponse(BaseModel):
    """Réponse API après logging"""
    status: str
    node_id: str
    timestamp: datetime
    message: Optional[str] = None
```

#### 4.1.5 Service Sentiment Analysis

```python
# services/sentiment.py
from transformers import pipeline
from functools import lru_cache
import logging

logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    """Analyseur de sentiment ML"""
    
    def __init__(self, model_name: str = "nlptown/bert-base-multilingual-uncased-sentiment"):
        """
        Initialise le modèle de sentiment
        
        Args:
            model_name: Nom modèle HuggingFace
                       (support multilingue FR/EN)
        """
        logger.info(f"Chargement modèle sentiment: {model_name}")
        self.analyzer = pipeline(
            "sentiment-analysis",
            model=model_name,
            device=-1  # CPU (GPU = 0)
        )
    
    @lru_cache(maxsize=1000)
    def analyze(self, texte: str) -> float:
        """
        Analyse sentiment d'un texte
        
        Args:
            texte: Texte à analyser
            
        Returns:
            float: Score sentiment [-1.0, 1.0]
                   -1.0 = très négatif
                    0.0 = neutre
                   +1.0 = très positif
        """
        if not texte or len(texte.strip()) == 0:
            return 0.0
        
        try:
            # Truncate si trop long (limite modèle)
            texte_truncated = texte[:512]
            
            # Analyse
            result = self.analyzer(texte_truncated)[0]
            
            # Normalisation score
            # Model retourne 1-5 stars, on normalise à [-1, 1]
            label = result['label']  # "1 star", "2 stars", etc.
            stars = int(label.split()[0])
            
            # Mapping: 1 star=-1.0, 2=-0.5, 3=0.0, 4=0.5, 5=1.0
            sentiment_score = (stars - 3) / 2.0
            
            logger.debug(f"Sentiment: {stars} stars -> {sentiment_score:.2f}")
            return sentiment_score
            
        except Exception as e:
            logger.error(f"Erreur analyse sentiment: {e}")
            return 0.0  # Fallback neutre

# Singleton global
_sentiment_analyzer = None

def get_sentiment_analyzer() -> SentimentAnalyzer:
    """Récupère instance singleton"""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = SentimentAnalyzer()
    return _sentiment_analyzer
```

#### 4.1.6 Service Embedding Generation

```python
# services/embedding.py
from sentence_transformers import SentenceTransformer
from functools import lru_cache
import numpy as np
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    """Générateur d'embeddings sémantiques"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialise le modèle d'embeddings
        
        Args:
            model_name: Nom modèle SentenceTransformers
                       (all-MiniLM-L6-v2 = 384 dims, rapide)
        """
        logger.info(f"Chargement modèle embeddings: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    @lru_cache(maxsize=1000)
    def generate(self, texte: str) -> List[float]:
        """
        Génère embedding pour un texte
        
        Args:
            texte: Texte à encoder
            
        Returns:
            List[float]: Vecteur embedding (384 dimensions)
        """
        if not texte or len(texte.strip()) == 0:
            return [0.0] * self.embedding_dim
        
        try:
            # Génération embedding
            embedding = self.model.encode(
                texte,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            
            # Conversion en liste Python (pour JSON/Neo4j)
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Erreur génération embedding: {e}")
            return [0.0] * self.embedding_dim

    @staticmethod
    def cosine_similarity(emb1: List[float], emb2: List[float]) -> float:
        """
        Calcule similarité cosinus entre deux embeddings
        
        Args:
            emb1, emb2: Vecteurs embeddings
            
        Returns:
            float: Similarité [0.0, 1.0]
        """
        v1 = np.array(emb1)
        v2 = np.array(emb2)
        
        dot_product = np.dot(v1, v2)
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(dot_product / (norm1 * norm2))

# Singleton global
_embedding_generator = None

def get_embedding_generator() -> EmbeddingGenerator:
    """Récupère instance singleton"""
    global _embedding_generator
    if _embedding_generator is None:
        _embedding_generator = EmbeddingGenerator()
    return _embedding_generator
```

#### 4.1.7 Service Persistence

```python
# services/persistence.py
from neo4j import GraphDatabase, Transaction
from typing import Dict, Any
from uuid import UUID
import logging
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class PersistenceService:
    """Service de persistance Neo4j"""
    
    def __init__(self, uri: str, user: str, password: str):
        """
        Initialise connexion Neo4j
        
        Args:
            uri: URI Neo4j (bolt://localhost:7687)
            user: Username
            password: Password
        """
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info(f"Connexion Neo4j établie: {uri}")
    
    def close(self):
        """Ferme connexion"""
        self.driver.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10)
    )
    def create_event_node(self, event: Event) -> str:
        """
        Crée un nœud Event dans Neo4j
        
        Args:
            event: Objet Event à persister
            
        Returns:
            str: ID du nœud créé
            
        Raises:
            Exception: Si échec après retries
        """
        with self.driver.session() as session:
            result = session.execute_write(
                self._create_event_tx,
                event
            )
            logger.info(f"Nœud créé: {result}")
            return result
    
    @staticmethod
    def _create_event_tx(tx: Transaction, event: Event) -> str:
        """
        Transaction de création nœud
        
        Args:
            tx: Transaction Neo4j
            event: Event à créer
            
        Returns:
            str: Node ID
        """
        query = """
        CREATE (e:Event {
            id: $id,
            timestamp_start: datetime($timestamp_start),
            timestamp_end: CASE WHEN $timestamp_end IS NOT NULL 
                               THEN datetime($timestamp_end) 
                               ELSE NULL END,
            agent: $agent,
            user_id: $user_id,
            session_id: $session_id,
            type: $type,
            
            // Contenu
            contenu_texte: $contenu_texte,
            contenu_sentiment: $contenu_sentiment,
            contenu_intensite: $contenu_intensite,
            contenu_tags: $contenu_tags,
            contenu_embeddings: $contenu_embeddings,
            
            // Contexte
            contexte_domaine: $contexte_domaine,
            contexte_phase: $contexte_phase,
            contexte_heure_journee: $contexte_heure_journee,
            contexte_jour_semaine: $contexte_jour_semaine
        })
        RETURN e.id as node_id
        """
        
        params = {
            "id": str(event.id),
            "timestamp_start": event.timestamp_start.isoformat(),
            "timestamp_end": event.timestamp_end.isoformat() if event.timestamp_end else None,
            "agent": event.agent,
            "user_id": event.user_id,
            "session_id": event.session_id,
            "type": event.type,
            
            # Contenu
            "contenu_texte": event.contenu.texte,
            "contenu_sentiment": event.contenu.sentiment_detecte,
            "contenu_intensite": event.contenu.intensite,
            "contenu_tags": event.contenu.tags,
            "contenu_embeddings": event.contenu.embeddings,
            
            # Contexte
            "contexte_domaine": event.contexte.domaine,
            "contexte_phase": event.contexte.phase,
            "contexte_heure_journee": event.contexte.heure_journee,
            "contexte_jour_semaine": event.contexte.jour_semaine
        }
        
        result = tx.run(query, **params)
        return result.single()["node_id"]
    
    def create_temporal_edge(self, node_id_from: str, node_id_to: str):
        """
        Crée arête TEMPOREL_SUITE entre deux nœuds
        
        Args:
            node_id_from: ID nœud source
            node_id_to: ID nœud cible
        """
        with self.driver.session() as session:
            session.execute_write(
                self._create_temporal_edge_tx,
                node_id_from,
                node_id_to
            )
    
    @staticmethod
    def _create_temporal_edge_tx(tx: Transaction, id_from: str, id_to: str):
        """Transaction création arête temporelle"""
        query = """
        MATCH (a:Event {id: $id_from})
        MATCH (b:Event {id: $id_to})
        MERGE (a)-[r:TEMPOREL_SUITE]->(b)
        RETURN r
        """
        tx.run(query, id_from=id_from, id_to=id_to)
```

#### 4.1.8 API Endpoints

```python
# main.py
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
from contextlib import asynccontextmanager

from models.event import EventCreate, Event, LogResponse
from services.sentiment import get_sentiment_analyzer
from services.embedding import get_embedding_generator
from services.persistence import PersistenceService
from utils.config import get_settings
from utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gestion lifecycle application"""
    logger.info("Démarrage Greffier API")
    
    # Startup
    settings = get_settings()
    app.state.persistence = PersistenceService(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password
    )
    
    # Warmup ML models
    get_sentiment_analyzer()
    get_embedding_generator()
    
    yield
    
    # Shutdown
    logger.info("Arrêt Greffier API")
    app.state.persistence.close()

# Application
app = FastAPI(
    title="Greffier API",
    description="Event Ingestion Service - Système MTE",
    version="1.0.0",
    lifespan=lifespan
)

# CORS (Phase 0 - permissif, à restreindre Phase 1)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
def get_persistence() -> PersistenceService:
    """Récupère service persistence"""
    return app.state.persistence

# Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "greffier",
        "version": "1.0.0"
    }

@app.post("/log_event", response_model=LogResponse)
async def log_event(
    event_create: EventCreate,
    persistence: PersistenceService = Depends(get_persistence)
):
    """
    Log un événement dans le système
    
    Args:
        event_create: Données événement
        persistence: Service persistence (injection)
        
    Returns:
        LogResponse: Confirmation avec node_id
        
    Raises:
        HTTPException: Si erreur validation ou persistance
    """
    try:
        # 1. Enrichissement automatique
        sentiment_analyzer = get_sentiment_analyzer()
        embedding_generator = get_embedding_generator()
        
        # Analyse sentiment
        sentiment_score = sentiment_analyzer.analyze(event_create.contenu.texte)
        event_create.contenu.sentiment_detecte = sentiment_score
        
        # Génération embeddings
        embeddings = embedding_generator.generate(event_create.contenu.texte)
        event_create.contenu.embeddings = embeddings
        
        # 2. Création objet Event complet
        event = Event(**event_create.dict())
        
        # 3. Persistance
        node_id = persistence.create_event_node(event)
        
        # 4. Réponse
        return LogResponse(
            status="success",
            node_id=node_id,
            timestamp=event.timestamp_start,
            message="Événement loggé avec succès"
        )
        
    except Exception as e:
        logger.error(f"Erreur log_event: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du logging: {str(e)}"
        )

@app.get("/stats")
async def get_stats(persistence: PersistenceService = Depends(get_persistence)):
    """
    Statistiques système
    
    Returns:
        Dict: Stats (nombre nœuds, arêtes, etc.)
    """
    # TODO: Implémenter requête Neo4j pour stats
    return {
        "total_events": 0,
        "total_edges": 0,
        "message": "Not implemented yet"
    }
```

#### 4.1.9 Configuration

```python
# utils/config.py
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Configuration application"""
    
    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_reload: bool = False
    
    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"
    
    # ML Models
    sentiment_model: str = "nlptown/bert-base-multilingual-uncased-sentiment"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/greffier.log"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

@lru_cache()
def get_settings() -> Settings:
    """Récupère configuration (cached)"""
    return Settings()
```

```bash
# .env.example
API_HOST=0.0.0.0
API_PORT=8000
API_RELOAD=true

NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_secure_password

SENTIMENT_MODEL=nlptown/bert-base-multilingual-uncased-sentiment
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

LOG_LEVEL=INFO
LOG_FILE=logs/greffier.log
```

---

### 4.2 LE JUGE V.E. (Correlation Detection Engine)

#### 4.2.1 Responsabilités

1. **Détection Co-occurrence Temporelle**
   - Identifie événements proches temporellement (fenêtre 2h)
   - Compte occurrences patterns récurrents
   - Crée arêtes `CORRELATION_OBSERVEE` si seuil dépassé

2. **Détection Similarité Sémantique**
   - Compare embeddings événements
   - Identifie contenus sémantiquement similaires
   - Lie événements thématiquement connexes

3. **Détection Séquençage**
   - Identifie patterns A précède B régulièrement
   - Détecte cycles récurrents
   - Crée arêtes séquentielles

4. **Maintenance Graph**
   - Cleanup arêtes faibles (decay)
   - Mise à jour force corrélations
   - Archivage patterns obsolètes

#### 4.2.2 Stack Technique

**Langage :** Python 3.11+

**Dépendances :**
```python
neo4j==5.14.0              # Driver database
numpy==1.24.3              # Calculs vectoriels
scikit-learn==1.3.2        # Métriques (cosine similarity)
schedule==1.2.0            # Job scheduling
python-dotenv==1.0.0       # Configuration
```

#### 4.2.3 Architecture Interne

```
Juge/
├── main.py                     # Point d'entrée batch
├── detectors/
│   ├── __init__.py
│   ├── temporal.py             # Détection co-occurrence temporelle
│   ├── semantic.py             # Détection similarité sémantique
│   └── sequential.py           # Détection séquençage
├── database/
│   ├── __init__.py
│   ├── connection.py           # Neo4j driver
│   └── queries.py              # Cypher queries
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── config.py
├── tests/
│   ├── test_temporal.py
│   ├── test_semantic.py
│   └── test_sequential.py
├── requirements.txt
└── Dockerfile
```

#### 4.2.4 Détecteur Co-occurrence Temporelle

```python
# detectors/temporal.py
from neo4j import GraphDatabase
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class TemporalCooccurrenceDetector:
    """Détecte co-occurrences temporelles"""
    
    def __init__(self, driver, window_hours: int = 2, threshold: int = 3):
        """
        Args:
            driver: Neo4j driver
            window_hours: Fenêtre temporelle (heures)
            threshold: Minimum occurrences pour créer arête
        """
        self.driver = driver
        self.window = timedelta(hours=window_hours)
        self.threshold = threshold
    
    def detect(self, user_id: str):
        """
        Détecte patterns co-occurrence pour un utilisateur
        
        Args:
            user_id: ID utilisateur à analyser
        """
        logger.info(f"Détection co-occurrence temporelle: user={user_id}")
        
        with self.driver.session() as session:
            # 1. Identifier patterns sentiment négatif → craving
            self._detect_negative_to_craving(session, user_id)
            
            # 2. Identifier patterns intervention → résultat
            self._detect_intervention_to_outcome(session, user_id)
            
            # 3. Identifier patterns génériques
            self._detect_generic_cooccurrence(session, user_id)
    
    def _detect_negative_to_craving(self, session, user_id: str):
        """
        Pattern: Sentiment négatif → Craving
        Ex: Stress travail (sentiment < -0.3) suivi de craving dans 2h
        """
        query = """
        MATCH (a:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..10]->(b:Event {user_id: $user_id})
        WHERE a.contenu_sentiment < -0.3
        AND (b.type = 'craving' OR b.contenu_texte CONTAINS 'craving' OR b.contenu_texte CONTAINS 'envie')
        AND duration.between(a.timestamp_start, b.timestamp_start) < duration({hours: $window_hours})
        WITH a.type as type_a, 
             b.type as type_b,
             count(*) as occurrences,
             collect(DISTINCT a.id)[0..5] as sample_a_ids,
             collect(DISTINCT b.id)[0..5] as sample_b_ids
        WHERE occurrences >= $threshold
        RETURN type_a, type_b, occurrences, sample_a_ids, sample_b_ids
        """
        
        result = session.run(
            query,
            user_id=user_id,
            window_hours=self.window.total_seconds() / 3600,
            threshold=self.threshold
        )
        
        for record in result:
            logger.info(
                f"Pattern détecté: {record['type_a']} → {record['type_b']} "
                f"({record['occurrences']} occurrences)"
            )
            
            # Créer arêtes corrélation
            self._create_correlation_edges(
                session,
                record['sample_a_ids'],
                record['sample_b_ids'],
                force=min(record['occurrences'] / 10.0, 1.0),
                occurrences=record['occurrences'],
                pattern_type='temporal_negative_to_craving'
            )
    
    def _detect_intervention_to_outcome(self, session, user_id: str):
        """
        Pattern: Intervention → Outcome
        Ex: Piano (intervention) suivi de sommeil amélioré
        """
        query = """
        MATCH (a:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..10]->(b:Event {user_id: $user_id})
        WHERE (a.type = 'intervention' OR a.type = 'external_event')
        AND b.contenu_sentiment IS NOT NULL
        AND duration.between(a.timestamp_start, b.timestamp_start) < duration({hours: $window_hours})
        WITH a.contenu_texte as intervention,
             b.contenu_sentiment as outcome_sentiment,
             count(*) as occurrences,
             avg(b.contenu_sentiment) as avg_sentiment,
             collect(DISTINCT a.id)[0..5] as sample_a_ids,
             collect(DISTINCT b.id)[0..5] as sample_b_ids
        WHERE occurrences >= $threshold
        AND abs(avg_sentiment) > 0.3  // Impact significatif
        RETURN intervention, avg_sentiment, occurrences, sample_a_ids, sample_b_ids
        """
        
        result = session.run(
            query,
            user_id=user_id,
            window_hours=self.window.total_seconds() / 3600,
            threshold=self.threshold
        )
        
        for record in result:
            logger.info(
                f"Pattern intervention: {record['intervention']} → "
                f"sentiment={record['avg_sentiment']:.2f} "
                f"({record['occurrences']} fois)"
            )
            
            self._create_correlation_edges(
                session,
                record['sample_a_ids'],
                record['sample_b_ids'],
                force=min(record['occurrences'] / 10.0, 1.0),
                occurrences=record['occurrences'],
                pattern_type='temporal_intervention_to_outcome'
            )
    
    def _detect_generic_cooccurrence(self, session, user_id: str):
        """
        Pattern générique: A suivi de B fréquemment
        """
        query = """
        MATCH (a:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..5]->(b:Event {user_id: $user_id})
        WHERE duration.between(a.timestamp_start, b.timestamp_start) < duration({hours: $window_hours})
        WITH a.type as type_a,
             b.type as type_b,
             count(*) as occurrences,
             collect(DISTINCT a.id)[0..5] as sample_a_ids,
             collect(DISTINCT b.id)[0..5] as sample_b_ids
        WHERE occurrences >= $threshold
        RETURN type_a, type_b, occurrences, sample_a_ids, sample_b_ids
        ORDER BY occurrences DESC
        LIMIT 10
        """
        
        result = session.run(
            query,
            user_id=user_id,
            window_hours=self.window.total_seconds() / 3600,
            threshold=self.threshold
        )
        
        for record in result:
            logger.info(
                f"Pattern générique: {record['type_a']} → {record['type_b']} "
                f"({record['occurrences']} fois)"
            )
            
            self._create_correlation_edges(
                session,
                record['sample_a_ids'],
                record['sample_b_ids'],
                force=min(record['occurrences'] / 10.0, 1.0),
                occurrences=record['occurrences'],
                pattern_type='temporal_generic'
            )
    
    def _create_correlation_edges(
        self,
        session,
        ids_a: list,
        ids_b: list,
        force: float,
        occurrences: int,
        pattern_type: str
    ):
        """
        Crée arêtes CORRELATION_OBSERVEE entre échantillons
        
        Args:
            session: Neo4j session
            ids_a: IDs nœuds source
            ids_b: IDs nœuds cible
            force: Force corrélation [0, 1]
            occurrences: Nombre total occurrences
            pattern_type: Type de pattern détecté
        """
        query = """
        UNWIND $ids_a as id_a
        UNWIND $ids_b as id_b
        MATCH (a:Event {id: id_a})
        MATCH (b:Event {id: id_b})
        WHERE NOT EXISTS((a)-[:CORRELATION_OBSERVEE]->(b))
        MERGE (a)-[r:CORRELATION_OBSERVEE]->(b)
        SET r.force = $force,
            r.occurrences = $occurrences,
            r.pattern_type = $pattern_type,
            r.created_at = datetime(),
            r.last_updated = datetime()
        """
        
        session.run(
            query,
            ids_a=ids_a,
            ids_b=ids_b,
            force=force,
            occurrences=occurrences,
            pattern_type=pattern_type
        )
```

#### 4.2.5 Détecteur Similarité Sémantique

```python
# detectors/semantic.py
from neo4j import GraphDatabase
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SemanticSimilarityDetector:
    """Détecte similarités sémantiques entre événements"""
    
    def __init__(self, driver, similarity_threshold: float = 0.75):
        """
        Args:
            driver: Neo4j driver
            similarity_threshold: Seuil similarité cosinus [0, 1]
        """
        self.driver = driver
        self.similarity_threshold = similarity_threshold
    
    def detect(self, user_id: str, lookback_days: int = 21):
        """
        Détecte similarités sémantiques
        
        Args:
            user_id: ID utilisateur
            lookback_days: Période analyse (jours)
        """
        logger.info(f"Détection similarité sémantique: user={user_id}")
        
        with self.driver.session() as session:
            # Récupérer événements récents avec embeddings
            events = self._get_events_with_embeddings(
                session,
                user_id,
                lookback_days
            )
            
            if len(events) < 2:
                logger.warning("Pas assez d'événements pour analyse sémantique")
                return
            
            # Calculer matrice similarité
            similarities = self._compute_similarity_matrix(events)
            
            # Créer arêtes pour similarités élevées
            self._create_similarity_edges(session, events, similarities)
    
    def _get_events_with_embeddings(
        self,
        session,
        user_id: str,
        lookback_days: int
    ) -> list:
        """
        Récupère événements avec embeddings
        
        Returns:
            List[Dict]: [{id, texte, embeddings, sentiment}, ...]
        """
        query = """
        MATCH (e:Event {user_id: $user_id})
        WHERE e.contenu_embeddings IS NOT NULL
        AND e.timestamp_start > datetime() - duration({days: $lookback_days})
        RETURN e.id as id,
               e.contenu_texte as texte,
               e.contenu_embeddings as embeddings,
               e.contenu_sentiment as sentiment,
               e.timestamp_start as timestamp
        ORDER BY e.timestamp_start DESC
        """
        
        result = session.run(
            query,
            user_id=user_id,
            lookback_days=lookback_days
        )
        
        return [dict(record) for record in result]
    
    def _compute_similarity_matrix(self, events: list) -> np.ndarray:
        """
        Calcule matrice similarité cosinus
        
        Args:
            events: Liste événements avec embeddings
            
        Returns:
            np.ndarray: Matrice similarité (n x n)
        """
        # Extraire embeddings
        embeddings = np.array([e['embeddings'] for e in events])
        
        # Calculer similarité cosinus
        similarity_matrix = cosine_similarity(embeddings)
        
        return similarity_matrix
    
    def _create_similarity_edges(
        self,
        session,
        events: list,
        similarities: np.ndarray
    ):
        """
        Crée arêtes CORRELATION_OBSERVEE pour similarités élevées
        
        Args:
            session: Neo4j session
            events: Liste événements
            similarities: Matrice similarité
        """
        edges_created = 0
        
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                similarity = similarities[i, j]
                
                if similarity >= self.similarity_threshold:
                    # Créer arête
                    query = """
                    MATCH (a:Event {id: $id_a})
                    MATCH (b:Event {id: $id_b})
                    WHERE NOT EXISTS((a)-[:CORRELATION_OBSERVEE]-(b))
                    MERGE (a)-[r:CORRELATION_OBSERVEE]-(b)
                    SET r.force = $similarity,
                        r.pattern_type = 'semantic_similarity',
                        r.created_at = datetime(),
                        r.last_updated = datetime()
                    """
                    
                    session.run(
                        query,
                        id_a=events[i]['id'],
                        id_b=events[j]['id'],
                        similarity=float(similarity)
                    )
                    
                    edges_created += 1
                    
                    logger.debug(
                        f"Similarité détectée: "
                        f"{events[i]['texte'][:50]} <-> "
                        f"{events[j]['texte'][:50]} "
                        f"(score: {similarity:.3f})"
                    )
        
        logger.info(f"Arêtes sémantiques créées: {edges_created}")
```

#### 4.2.6 Détecteur Séquençage

```python
# detectors/sequential.py
from neo4j import GraphDatabase
from datetime import timedelta
import logging

logger = logging.getLogger(__name__)

class SequentialPatternDetector:
    """Détecte patterns séquentiels (A précède systématiquement B)"""
    
    def __init__(self, driver, max_delay_hours: int = 12, threshold: int = 3):
        """
        Args:
            driver: Neo4j driver
            max_delay_hours: Délai maximum entre A et B
            threshold: Minimum occurrences
        """
        self.driver = driver
        self.max_delay = timedelta(hours=max_delay_hours)
        self.threshold = threshold
    
    def detect(self, user_id: str):
        """
        Détecte patterns séquentiels
        
        Args:
            user_id: ID utilisateur
        """
        logger.info(f"Détection patterns séquentiels: user={user_id}")
        
        with self.driver.session() as session:
            # Patterns spécifiques connus
            self._detect_insomnia_to_fatigue(session, user_id)
            self._detect_craving_to_intervention(session, user_id)
            
            # Patterns génériques
            self._detect_generic_sequences(session, user_id)
    
    def _detect_insomnia_to_fatigue(self, session, user_id: str):
        """
        Pattern: Insomnie (nuit) → Fatigue (matin suivant)
        """
        query = """
        MATCH (a:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..10]->(b:Event {user_id: $user_id})
        WHERE (a.type = 'insomnie' OR a.contenu_texte CONTAINS 'insomnie' OR a.contenu_texte CONTAINS 'dormi')
        AND (b.type = 'fatigue' OR b.contenu_texte CONTAINS 'fatigué' OR b.contenu_texte CONTAINS 'épuisé')
        AND duration.between(a.timestamp_start, b.timestamp_start) < duration({hours: $max_delay_hours})
        AND b.timestamp_start > a.timestamp_start
        WITH count(*) as occurrences,
             collect(DISTINCT a.id)[0..5] as sample_a_ids,
             collect(DISTINCT b.id)[0..5] as sample_b_ids
        WHERE occurrences >= $threshold
        RETURN occurrences, sample_a_ids, sample_b_ids
        """
        
        result = session.run(
            query,
            user_id=user_id,
            max_delay_hours=self.max_delay.total_seconds() / 3600,
            threshold=self.threshold
        )
        
        for record in result:
            logger.info(
                f"Pattern séquentiel: Insomnie → Fatigue "
                f"({record['occurrences']} occurrences)"
            )
            
            self._create_sequential_edges(
                session,
                record['sample_a_ids'],
                record['sample_b_ids'],
                force=min(record['occurrences'] / 10.0, 1.0),
                occurrences=record['occurrences'],
                pattern_type='sequential_insomnia_fatigue'
            )
    
    def _detect_craving_to_intervention(self, session, user_id: str):
        """
        Pattern: Craving → Intervention (utilisateur agit)
        """
        query = """
        MATCH (a:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..5]->(b:Event {user_id: $user_id})
        WHERE (a.type = 'craving' OR a.contenu_texte CONTAINS 'craving' OR a.contenu_texte CONTAINS 'envie')
        AND (b.type = 'intervention' OR b.type = 'external_event')
        AND duration.between(a.timestamp_start, b.timestamp_start) < duration({hours: $max_delay_hours})
        WITH a.contenu_texte as craving_desc,
             b.contenu_texte as intervention,
             count(*) as occurrences,
             collect(DISTINCT a.id)[0..5] as sample_a_ids,
             collect(DISTINCT b.id)[0..5] as sample_b_ids
        WHERE occurrences >= $threshold
        RETURN craving_desc, intervention, occurrences, sample_a_ids, sample_b_ids
        """
        
        result = session.run(
            query,
            user_id=user_id,
            max_delay_hours=self.max_delay.total_seconds() / 3600,
            threshold=self.threshold
        )
        
        for record in result:
            logger.info(
                f"Pattern séquentiel: Craving → {record['intervention']} "
                f"({record['occurrences']} fois)"
            )
            
            self._create_sequential_edges(
                session,
                record['sample_a_ids'],
                record['sample_b_ids'],
                force=min(record['occurrences'] / 10.0, 1.0),
                occurrences=record['occurrences'],
                pattern_type='sequential_craving_intervention'
            )
    
    def _detect_generic_sequences(self, session, user_id: str):
        """
        Patterns séquentiels génériques
        """
        query = """
        MATCH (a:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..10]->(b:Event {user_id: $user_id})
        WHERE duration.between(a.timestamp_start, b.timestamp_start) < duration({hours: $max_delay_hours})
        WITH a.type as type_a,
             b.type as type_b,
             avg(duration.between(a.timestamp_start, b.timestamp_start).minutes) as avg_delay_minutes,
             count(*) as occurrences,
             collect(DISTINCT a.id)[0..5] as sample_a_ids,
             collect(DISTINCT b.id)[0..5] as sample_b_ids
        WHERE occurrences >= $threshold
        RETURN type_a, type_b, avg_delay_minutes, occurrences, sample_a_ids, sample_b_ids
        ORDER BY occurrences DESC
        LIMIT 10
        """
        
        result = session.run(
            query,
            user_id=user_id,
            max_delay_hours=self.max_delay.total_seconds() / 3600,
            threshold=self.threshold
        )
        
        for record in result:
            logger.info(
                f"Pattern séquentiel: {record['type_a']} → {record['type_b']} "
                f"(délai moyen: {record['avg_delay_minutes']:.1f}min, "
                f"{record['occurrences']} fois)"
            )
            
            self._create_sequential_edges(
                session,
                record['sample_a_ids'],
                record['sample_b_ids'],
                force=min(record['occurrences'] / 10.0, 1.0),
                occurrences=record['occurrences'],
                pattern_type='sequential_generic'
            )
    
    def _create_sequential_edges(
        self,
        session,
        ids_a: list,
        ids_b: list,
        force: float,
        occurrences: int,
        pattern_type: str
    ):
        """Crée arêtes CORRELATION_OBSERVEE pour séquences"""
        query = """
        UNWIND $ids_a as id_a
        UNWIND $ids_b as id_b
        MATCH (a:Event {id: id_a})
        MATCH (b:Event {id: id_b})
        WHERE NOT EXISTS((a)-[:CORRELATION_OBSERVEE]->(b))
        MERGE (a)-[r:CORRELATION_OBSERVEE]->(b)
        SET r.force = $force,
            r.occurrences = $occurrences,
            r.pattern_type = $pattern_type,
            r.created_at = datetime(),
            r.last_updated = datetime()
        """
        
        session.run(
            query,
            ids_a=ids_a,
            ids_b=ids_b,
            force=force,
            occurrences=occurrences,
            pattern_type=pattern_type
        )
```

#### 4.2.7 Script Principal Batch

```python
# main.py
import schedule
import time
import logging
from neo4j import GraphDatabase
from datetime import datetime

from detectors.temporal import TemporalCooccurrenceDetector
from detectors.semantic import SemanticSimilarityDetector
from detectors.sequential import SequentialPatternDetector
from utils.config import get_settings
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def run_correlation_detection():
    """
    Exécute détection corrélations
    """
    logger.info("=== DÉBUT BATCH DÉTECTION CORRÉLATIONS ===")
    start_time = datetime.now()
    
    try:
        settings = get_settings()
        driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
        # Récupérer liste utilisateurs
        with driver.session() as session:
            result = session.run("MATCH (e:Event) RETURN DISTINCT e.user_id as user_id")
            user_ids = [record['user_id'] for record in result]
        
        logger.info(f"Utilisateurs à analyser: {len(user_ids)}")
        
        # Détecteurs
        temporal_detector = TemporalCooccurrenceDetector(
            driver,
            window_hours=settings.temporal_window_hours,
            threshold=settings.correlation_threshold
        )
        
        semantic_detector = SemanticSimilarityDetector(
            driver,
            similarity_threshold=settings.semantic_similarity_threshold
        )
        
        sequential_detector = SequentialPatternDetector(
            driver,
            max_delay_hours=settings.sequential_max_delay_hours,
            threshold=settings.correlation_threshold
        )
        
        # Analyse par utilisateur
        for user_id in user_ids:
            logger.info(f"Analyse utilisateur: {user_id}")
            
            try:
                temporal_detector.detect(user_id)
                semantic_detector.detect(user_id)
                sequential_detector.detect(user_id)
            except Exception as e:
                logger.error(f"Erreur analyse {user_id}: {e}", exc_info=True)
        
        driver.close()
        
        duration = (datetime.now() - start_time).total_seconds()
        logger.info(f"=== FIN BATCH (durée: {duration:.1f}s) ===")
        
    except Exception as e:
        logger.error(f"Erreur batch: {e}", exc_info=True)
        raise

def main():
    """Point d'entrée principal"""
    logger.info("Démarrage Juge V.E. - Batch Service")
    
    settings = get_settings()
    
    # Schedule batch (3AM par défaut)
    schedule.every().day.at(settings.batch_schedule_time).do(run_correlation_detection)
    
    logger.info(f"Batch schedulé: {settings.batch_schedule_time}")
    
    # Exécution immédiate pour test
    if settings.run_on_startup:
        logger.info("Exécution immédiate (run_on_startup=true)")
        run_correlation_detection()
    
    # Boucle schedule
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check chaque minute

if __name__ == "__main__":
    main()
```

---

### 4.3 LE MÉMORIALISTE V.E. (Context Enrichment Engine)

#### 4.3.1 Responsabilités

1. **Récupération Contexte Temporel**
   - Derniers N événements utilisateur
   - Historique conversation session actuelle
   - Événements journée en cours

2. **Récupération Contexte Sémantique**
   - Recherche similarité embeddings
   - Événements thématiquement proches
   - Références historiques pertinentes

3. **Récupération Patterns Corrélation**
   - Arêtes CORRELATION_OBSERVEE pertinentes
   - Patterns récurrents actifs
   - Hypothèses causales potentielles

4. **Construction Prompt Empathique**
   - Format contexte pour LLM
   - Instructions empathiques (ne pas conclure)
   - Invitation validation utilisateur

#### 4.3.2 Stack Technique

**Langage :** Python 3.11+

**Dépendances :**
```python
neo4j==5.14.0
numpy==1.24.3
scikit-learn==1.3.2        # Cosine similarity
sentence-transformers==2.2.2
anthropic==0.7.0           # Claude API (intégration)
```

#### 4.3.3 Architecture Interne

```
Memorialiste/
├── main.py                     # Service principal
├── retrievers/
│   ├── __init__.py
│   ├── temporal.py             # Récupération temporelle
│   ├── semantic.py             # Récupération sémantique
│   └── correlation.py          # Récupération patterns
├── formatters/
│   ├── __init__.py
│   └── prompt.py               # Construction prompt empathique
├── database/
│   ├── __init__.py
│   └── connection.py
├── utils/
│   ├── __init__.py
│   ├── logger.py
│   └── config.py
└── tests/
    └── test_enrichment.py
```

#### 4.3.4 Retriever Temporel

```python
# retrievers/temporal.py
from neo4j import GraphDatabase
from typing import List, Dict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class TemporalRetriever:
    """Récupère contexte temporel"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_recent_events(
        self,
        user_id: str,
        limit: int = 5,
        session_id: str = None
    ) -> List[Dict]:
        """
        Récupère événements récents
        
        Args:
            user_id: ID utilisateur
            limit: Nombre max événements
            session_id: Filtrer par session (optionnel)
            
        Returns:
            List[Dict]: Événements triés chronologiquement (desc)
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Event {user_id: $user_id})
            WHERE ($session_id IS NULL OR e.session_id = $session_id)
            RETURN e.id as id,
                   e.timestamp_start as timestamp,
                   e.type as type,
                   e.contenu_texte as texte,
                   e.contenu_sentiment as sentiment,
                   e.contenu_tags as tags
            ORDER BY e.timestamp_start DESC
            LIMIT $limit
            """
            
            result = session.run(
                query,
                user_id=user_id,
                session_id=session_id,
                limit=limit
            )
            
            return [dict(record) for record in result]
    
    def get_today_events(self, user_id: str) -> List[Dict]:
        """
        Récupère événements d'aujourd'hui
        
        Args:
            user_id: ID utilisateur
            
        Returns:
            List[Dict]: Événements du jour
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Event {user_id: $user_id})
            WHERE date(e.timestamp_start) = date()
            RETURN e.id as id,
                   e.timestamp_start as timestamp,
                   e.type as type,
                   e.contenu_texte as texte,
                   e.contenu_sentiment as sentiment
            ORDER BY e.timestamp_start ASC
            """
            
            result = session.run(query, user_id=user_id)
            return [dict(record) for record in result]
    
    def get_lookback_events(
        self,
        user_id: str,
        days: int = 3,
        event_types: List[str] = None
    ) -> List[Dict]:
        """
        Récupère événements sur période
        
        Args:
            user_id: ID utilisateur
            days: Nombre jours lookback
            event_types: Filtrer par types (optionnel)
            
        Returns:
            List[Dict]: Événements période
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Event {user_id: $user_id})
            WHERE e.timestamp_start > datetime() - duration({days: $days})
            AND ($event_types IS NULL OR e.type IN $event_types)
            RETURN e.id as id,
                   e.timestamp_start as timestamp,
                   e.type as type,
                   e.contenu_texte as texte,
                   e.contenu_sentiment as sentiment,
                   e.contenu_tags as tags
            ORDER BY e.timestamp_start DESC
            """
            
            result = session.run(
                query,
                user_id=user_id,
                days=days,
                event_types=event_types
            )
            
            return [dict(record) for record in result]
```

#### 4.3.5 Retriever Sémantique

```python
# retrievers/semantic.py
from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class SemanticRetriever:
    """Récupère contexte sémantiquement similaire"""
    
    def __init__(self, driver, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.driver = driver
        self.model = SentenceTransformer(model_name)
    
    def get_similar_events(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 3,
        lookback_days: int = 21,
        similarity_threshold: float = 0.7
    ) -> List[Dict]:
        """
        Récupère événements sémantiquement similaires
        
        Args:
            user_id: ID utilisateur
            query_text: Texte requête (message actuel)
            top_k: Nombre résultats max
            lookback_days: Période recherche
            similarity_threshold: Seuil similarité minimum
            
        Returns:
            List[Dict]: Événements similaires avec score
        """
        # Générer embedding requête
        query_embedding = self.model.encode(query_text, convert_to_numpy=True)
        
        # Récupérer événements candidats
        with self.driver.session() as session:
            query = """
            MATCH (e:Event {user_id: $user_id})
            WHERE e.contenu_embeddings IS NOT NULL
            AND e.timestamp_start > datetime() - duration({days: $lookback_days})
            AND e.contenu_texte <> $query_text
            RETURN e.id as id,
                   e.timestamp_start as timestamp,
                   e.type as type,
                   e.contenu_texte as texte,
                   e.contenu_sentiment as sentiment,
                   e.contenu_embeddings as embeddings
            """
            
            result = session.run(
                query,
                user_id=user_id,
                lookback_days=lookback_days,
                query_text=query_text
            )
            
            candidates = [dict(record) for record in result]
        
        if not candidates:
            return []
        
        # Calculer similarités
        candidate_embeddings = np.array([c['embeddings'] for c in candidates])
        similarities = cosine_similarity(
            query_embedding.reshape(1, -1),
            candidate_embeddings
        )[0]
        
        # Ajouter scores et filtrer
        similar_events = []
        for i, candidate in enumerate(candidates):
            similarity = float(similarities[i])
            if similarity >= similarity_threshold:
                candidate['similarity'] = similarity
                similar_events.append(candidate)
        
        # Trier par similarité et limiter
        similar_events.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_events[:top_k]
    
    def get_similar_events_from_embedding(
        self,
        user_id: str,
        query_embedding: List[float],
        top_k: int = 3,
        lookback_days: int = 21
    ) -> List[Dict]:
        """
        Récupère événements similaires depuis embedding
        (utile si embedding déjà calculé)
        """
        with self.driver.session() as session:
            query = """
            MATCH (e:Event {user_id: $user_id})
            WHERE e.contenu_embeddings IS NOT NULL
            AND e.timestamp_start > datetime() - duration({days: $lookback_days})
            RETURN e.id as id,
                   e.timestamp_start as timestamp,
                   e.type as type,
                   e.contenu_texte as texte,
                   e.contenu_sentiment as sentiment,
                   e.contenu_embeddings as embeddings
            """
            
            result = session.run(
                query,
                user_id=user_id,
                lookback_days=lookback_days
            )
            
            candidates = [dict(record) for record in result]
        
        if not candidates:
            return []
        
        # Calcul similarités
        query_emb = np.array(query_embedding).reshape(1, -1)
        candidate_embeddings = np.array([c['embeddings'] for c in candidates])
        similarities = cosine_similarity(query_emb, candidate_embeddings)[0]
        
        # Ajouter scores
        for i, candidate in enumerate(candidates):
            candidate['similarity'] = float(similarities[i])
        
        # Trier et limiter
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        return candidates[:top_k]
```

#### 4.3.6 Retriever Corrélations

```python
# retrievers/correlation.py
from neo4j import GraphDatabase
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class CorrelationRetriever:
    """Récupère patterns de corrélation pertinents"""
    
    def __init__(self, driver):
        self.driver = driver
    
    def get_relevant_correlations(
        self,
        user_id: str,
        current_event_type: str = None,
        current_tags: List[str] = None,
        min_force: float = 0.6,
        top_k: int = 3
    ) -> List[Dict]:
        """
        Récupère corrélations pertinentes au contexte actuel
        
        Args:
            user_id: ID utilisateur
            current_event_type: Type événement actuel (optionnel)
            current_tags: Tags événement actuel (optionnel)
            min_force: Force minimum corrélation
            top_k: Nombre max résultats
            
        Returns:
            List[Dict]: Corrélations [{type_a, texte_a, type_b, texte_b, force, ...}]
        """
        with self.driver.session() as session:
            # Requête basique (toutes corrélations fortes)
            if not current_event_type and not current_tags:
                query = """
                MATCH (a:Event {user_id: $user_id})-[r:CORRELATION_OBSERVEE]->(b:Event)
                WHERE r.force >= $min_force
                RETURN DISTINCT
                       a.type as type_a,
                       a.contenu_texte as texte_a,
                       b.type as type_b,
                       b.contenu_texte as texte_b,
                       r.force as force,
                       r.occurrences as occurrences,
                       r.pattern_type as pattern_type
                ORDER BY r.force DESC
                LIMIT $top_k
                """
                
                result = session.run(
                    query,
                    user_id=user_id,
                    min_force=min_force,
                    top_k=top_k
                )
            
            # Requête ciblée (pertinence au contexte)
            else:
                query = """
                MATCH (a:Event {user_id: $user_id})-[r:CORRELATION_OBSERVEE]->(b:Event)
                WHERE r.force >= $min_force
                AND (
                    ($current_type IS NOT NULL AND (a.type = $current_type OR b.type = $current_type))
                    OR
                    ($current_tags IS NOT NULL AND (
                        any(tag IN $current_tags WHERE tag IN a.contenu_tags) OR
                        any(tag IN $current_tags WHERE tag IN b.contenu_tags)
                    ))
                )
                RETURN DISTINCT
                       a.type as type_a,
                       a.contenu_texte as texte_a,
                       b.type as type_b,
                       b.contenu_texte as texte_b,
                       r.force as force,
                       r.occurrences as occurrences,
                       r.pattern_type as pattern_type
                ORDER BY r.force DESC
                LIMIT $top_k
                """
                
                result = session.run(
                    query,
                    user_id=user_id,
                    current_type=current_event_type,
                    current_tags=current_tags,
                    min_force=min_force,
                    top_k=top_k
                )
            
            return [dict(record) for record in result]
    
    def get_correlations_from_event_id(
        self,
        event_id: str,
        direction: str = "outgoing",
        min_force: float = 0.6
    ) -> List[Dict]:
        """
        Récupère corrélations depuis un événement spécifique
        
        Args:
            event_id: ID événement source
            direction: "outgoing", "incoming", ou "both"
            min_force: Force minimum
            
        Returns:
            List[Dict]: Corrélations
        """
        with self.driver.session() as session:
            if direction == "outgoing":
                query = """
                MATCH (a:Event {id: $event_id})-[r:CORRELATION_OBSERVEE]->(b:Event)
                WHERE r.force >= $min_force
                RETURN b.id as related_event_id,
                       b.type as type,
                       b.contenu_texte as texte,
                       r.force as force,
                       r.pattern_type as pattern_type,
                       'outgoing' as direction
                ORDER BY r.force DESC
                """
            elif direction == "incoming":
                query = """
                MATCH (a:Event)-[r:CORRELATION_OBSERVEE]->(b:Event {id: $event_id})
                WHERE r.force >= $min_force
                RETURN a.id as related_event_id,
                       a.type as type,
                       a.contenu_texte as texte,
                       r.force as force,
                       r.pattern_type as pattern_type,
                       'incoming' as direction
                ORDER BY r.force DESC
                """
            else:  # both
                query = """
                MATCH (a:Event {id: $event_id})-[r:CORRELATION_OBSERVEE]-(b:Event)
                WHERE r.force >= $min_force
                RETURN b.id as related_event_id,
                       b.type as type,
                       b.contenu_texte as texte,
                       r.force as force,
                       r.pattern_type as pattern_type,
                       CASE
                           WHEN startNode(r) = a THEN 'outgoing'
                           ELSE 'incoming'
                       END as direction
                ORDER BY r.force DESC
                """
            
            result = session.run(
                query,
                event_id=event_id,
                min_force=min_force
            )
            
            return [dict(record) for record in result]
```

#### 4.3.7 Formatter Prompt Empathique

```python
# formatters/prompt.py
from typing import List, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class EmpatheticPromptFormatter:
    """Construit prompts empathiques enrichis"""
    
    @staticmethod
    def format_enriched_prompt(
        user_message: str,
        temporal_events: List[Dict],
        semantic_events: List[Dict],
        correlations: List[Dict],
        current_sentiment: float = None
    ) -> str:
        """
        Construit prompt enrichi avec contexte mémoriel
        
        Args:
            user_message: Message utilisateur actuel
            temporal_events: Événements récents
            semantic_events: Événements similaires
            correlations: Patterns corrélation
            current_sentiment: Sentiment message actuel (optionnel)
            
        Returns:
            str: Prompt enrichi formaté
        """
        prompt_parts = []
        
        # Header
        prompt_parts.append("=== CONTEXTE MÉMORIEL (Pour Réponse Empathique) ===\n")
        
        # Message actuel
        prompt_parts.append("MESSAGE UTILISATEUR ACTUEL :")
        prompt_parts.append(f'"{user_message}"')
        if current_sentiment is not None:
            prompt_parts.append(f"Sentiment détecté : {current_sentiment:.2f}")
        prompt_parts.append("")
        
        # Contexte temporel
        if temporal_events:
            prompt_parts.append("CONTINUITÉ TEMPORELLE (Événements récents) :")
            for event in temporal_events:
                timestamp = event['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                # Formatage relatif (ex: "il y a 2 heures")
                time_ago = EmpatheticPromptFormatter._format_time_ago(timestamp)
                
                sentiment_str = ""
                if event.get('sentiment') is not None:
                    sentiment_str = f" [sentiment: {event['sentiment']:.2f}]"
                
                prompt_parts.append(
                    f"  • {time_ago} : {event['texte'][:100]}{sentiment_str}"
                )
            prompt_parts.append("")
        
        # Contexte sémantique
        if semantic_events:
            prompt_parts.append("CONNEXIONS SÉMANTIQUES (Sujets similaires passés) :")
            for event in semantic_events:
                timestamp = event['timestamp']
                if isinstance(timestamp, str):
                    timestamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                
                time_ago = EmpatheticPromptFormatter._format_time_ago(timestamp)
                similarity = event.get('similarity', 0)
                
                prompt_parts.append(
                    f"  • {time_ago} (similarité: {similarity:.2f}) : "
                    f"{event['texte'][:100]}"
                )
            prompt_parts.append("")
        
        # Patterns corrélation
        if correlations:
            prompt_parts.append("PATTERNS OBSERVÉS (Corrélations récurrentes) :")
            for corr in correlations:
                pattern_desc = EmpatheticPromptFormatter._format_correlation(corr)
                prompt_parts.append(f"  • {pattern_desc}")
            prompt_parts.append("")
        
        # Instructions empathiques
        prompt_parts.append("=== INSTRUCTIONS POUR TA RÉPONSE ===")
        prompt_parts.append("")
        prompt_parts.append("1. UTILISE CES SOUVENIRS pour montrer continuité conversationnelle")
        prompt_parts.append("   - Référence événements passés de manière naturelle")
        prompt_parts.append("   - Montre que tu te souviens et comprends le contexte")
        prompt_parts.append("")
        prompt_parts.append("2. NE FAIS PAS de conclusions causales définitives")
        prompt_parts.append("   - Évite : \"X cause Y\" ou \"X est la raison de Y\"")
        prompt_parts.append("   - Préfère : \"J'ai remarqué que... Est-ce lié ?\"")
        prompt_parts.append("")
        prompt_parts.append("3. PRÉSENTE les observations comme QUESTIONS EMPATHIQUES")
        prompt_parts.append("   - \"J'ai remarqué que la dernière fois que tu as mentionné [X], tu as aussi [Y]. Est-ce un pattern pour toi ?\"")
        prompt_parts.append("   - \"Tu m'as parlé de [X] il y a [T]. Comment ça évolue ?\"")
        prompt_parts.append("")
        prompt_parts.append("4. VALIDE/INVALIDE avec l'utilisateur, ne CONCLUS pas seul")
        prompt_parts.append("   - Laisse l'utilisateur confirmer ou infirmer tes observations")
        prompt_parts.append("   - Ton rôle = SE SOUVENIR et REFLÉTER, pas diagnostiquer")
        prompt_parts.append("")
        prompt_parts.append("5. TON EMPATHIQUE, naturel, authentique")
        prompt_parts.append("   - Pas de langage clinique ou robotique")
        prompt_parts.append("   - Montre que tu te soucies et que tu te souviens")
        prompt_parts.append("")
        
        return "\n".join(prompt_parts)
    
    @staticmethod
    def _format_time_ago(timestamp: datetime) -> str:
        """
        Formate durée relative
        
        Args:
            timestamp: Datetime événement
            
        Returns:
            str: "il y a X heures/jours/..."
        """
        now = datetime.now(timestamp.tzinfo)
        delta = now - timestamp
        
        seconds = delta.total_seconds()
        
        if seconds < 60:
            return "à l'instant"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"il y a {minutes} minute{'s' if minutes > 1 else ''}"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"il y a {hours} heure{'s' if hours > 1 else ''}"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"il y a {days} jour{'s' if days > 1 else ''}"
        else:
            weeks = int(seconds / 604800)
            return f"il y a {weeks} semaine{'s' if weeks > 1 else ''}"
    
    @staticmethod
    def _format_correlation(corr: Dict) -> str:
        """
        Formate description corrélation
        
        Args:
            corr: Dictionnaire corrélation
            
        Returns:
            str: Description formatée
        """
        type_a = corr.get('type_a', 'événement')
        type_b = corr.get('type_b', 'événement')
        texte_a = corr.get('texte_a', '')[:50]
        texte_b = corr.get('texte_b', '')[:50]
        force = corr.get('force', 0)
        occurrences = corr.get('occurrences', 0)
        pattern_type = corr.get('pattern_type', '')
        
        # Simplification pattern_type pour affichage
        pattern_label = {
            'temporal_negative_to_craving': 'temporel',
            'temporal_intervention_to_outcome': 'intervention → résultat',
            'semantic_similarity': 'similarité thématique',
            'sequential_insomnia_fatigue': 'séquence',
            'temporal_generic': 'temporel'
        }.get(pattern_type, pattern_type)
        
        return (
            f"Pattern {pattern_label}: \"{texte_a}\" souvent suivi de \"{texte_b}\" "
            f"(observé {occurrences} fois, force: {force:.2f})"
        )
```

#### 4.3.8 Service Principal Mémorialiste

```python
# main.py
from neo4j import GraphDatabase
from typing import Dict
import logging

from retrievers.temporal import TemporalRetriever
from retrievers.semantic import SemanticRetriever
from retrievers.correlation import CorrelationRetriever
from formatters.prompt import EmpatheticPromptFormatter
from utils.config import get_settings
from utils.logger import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

class MemorialisteService:
    """Service d'enrichissement contexte mémoriel"""
    
    def __init__(self):
        settings = get_settings()
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
        self.temporal_retriever = TemporalRetriever(self.driver)
        self.semantic_retriever = SemanticRetriever(
            self.driver,
            model_name=settings.embedding_model
        )
        self.correlation_retriever = CorrelationRetriever(self.driver)
        self.formatter = EmpatheticPromptFormatter()
    
    def enrich_context(
        self,
        user_id: str,
        user_message: str,
        session_id: str = None,
        current_sentiment: float = None
    ) -> str:
        """
        Enrichit contexte conversationnel
        
        Args:
            user_id: ID utilisateur
            user_message: Message utilisateur actuel
            session_id: ID session (optionnel)
            current_sentiment: Sentiment message (optionnel)
            
        Returns:
            str: Prompt enrichi pour LLM
        """
        logger.info(f"Enrichissement contexte: user={user_id}, session={session_id}")
        
        try:
            # 1. Récupération temporelle
            temporal_events = self.temporal_retriever.get_recent_events(
                user_id=user_id,
                limit=5,
                session_id=session_id
            )
            logger.debug(f"Événements temporels: {len(temporal_events)}")
            
            # 2. Récupération sémantique
            semantic_events = self.semantic_retriever.get_similar_events(
                user_id=user_id,
                query_text=user_message,
                top_k=3,
                lookback_days=21,
                similarity_threshold=0.7
            )
            logger.debug(f"Événements sémantiques: {len(semantic_events)}")
            
            # 3. Récupération corrélations
            # TODO: Extraire type/tags du message actuel pour ciblage
            correlations = self.correlation_retriever.get_relevant_correlations(
                user_id=user_id,
                min_force=0.6,
                top_k=3
            )
            logger.debug(f"Corrélations: {len(correlations)}")
            
            # 4. Construction prompt enrichi
            enriched_prompt = self.formatter.format_enriched_prompt(
                user_message=user_message,
                temporal_events=temporal_events,
                semantic_events=semantic_events,
                correlations=correlations,
                current_sentiment=current_sentiment
            )
            
            logger.info("Contexte enrichi avec succès")
            return enriched_prompt
            
        except Exception as e:
            logger.error(f"Erreur enrichissement: {e}", exc_info=True)
            # Fallback: retour message original
            return user_message
    
    def close(self):
        """Ferme connexions"""
        self.driver.close()

# Instance singleton
_memorialiste_service = None

def get_memorialiste() -> MemorialisteService:
    """Récupère instance singleton"""
    global _memorialiste_service
    if _memorialiste_service is None:
        _memorialiste_service = MemorialisteService()
    return _memorialiste_service
```

**ARCHITECTURE_PART2.md**

---

# ARCHITECTURE LOGICIELLE - SYSTÈME MTE (PARTIE 2)

**Version :** 1.0.0 (Phase 0 - V.E.)  
**Date :** 26 Octobre 2025  
**Suite de :** ARCHITECTURE.md

---

## 5. MODÈLE DE DONNÉES

### 5.1 Vue d'Ensemble

Le système utilise un **modèle de graphe orienté** (Neo4j) pour capturer la temporalité et les relations entre événements.

**Avantages modèle graphe :**
- Représentation naturelle relations causales/temporelles
- Requêtes de traversée efficaces (patterns, chemins)
- Flexibilité schéma (ajout propriétés sans migration)
- Visualisation intuitive

**Composants :**
- **Nœuds** : Événements (Event)
- **Arêtes** : Relations temporelles et corrélations
- **Propriétés** : Métadonnées événements/relations

### 5.2 Schéma Neo4j Complet

```cypher
// ============================================
// CONTRAINTES & INDEX
// ============================================

// Contrainte unicité ID
CREATE CONSTRAINT event_id_unique IF NOT EXISTS
FOR (e:Event) REQUIRE e.id IS UNIQUE;

// Index temporels (performance requêtes temporelles)
CREATE INDEX event_timestamp IF NOT EXISTS
FOR (e:Event) ON (e.timestamp_start);

CREATE INDEX event_timestamp_end IF NOT EXISTS
FOR (e:Event) ON (e.timestamp_end);

// Index utilisateur (isolation données multi-users)
CREATE INDEX event_user_id IF NOT EXISTS
FOR (e:Event) ON (e.user_id);

// Index session (requêtes conversationnelles)
CREATE INDEX event_session_id IF NOT EXISTS
FOR (e:Event) ON (e.session_id);

// Index type événement (filtrage par type)
CREATE INDEX event_type IF NOT EXISTS
FOR (e:Event) ON (e.type);

// Index sentiment (requêtes patterns émotionnels)
CREATE INDEX event_sentiment IF NOT EXISTS
FOR (e:Event) ON (e.contenu_sentiment);

// Index domaine/phase (filtrage contexte)
CREATE INDEX event_domaine IF NOT EXISTS
FOR (e:Event) ON (e.contexte_domaine);

CREATE INDEX event_phase IF NOT EXISTS
FOR (e:Event) ON (e.contexte_phase);

// ============================================
// NŒUDS : EVENT
// ============================================

// Structure complète d'un nœud Event
(:Event {
    // Identifiants
    id: String (UUID),                          // UNIQUE, PRIMARY KEY
    timestamp_start: DateTime,                  // ISO 8601
    timestamp_end: DateTime?,                   // Optionnel (événements durée)
    
    // Acteurs
    agent: String,                              // "rodin", "lumi", "nova"
    user_id: String,                            // Hash anonymisé
    session_id: String,                         // UUID session conversationnelle
    
    // Type événement
    type: String,                               // Enum (voir 5.3)
    
    // Contenu
    contenu_texte: String,                      // Texte événement (max 10000 chars)
    contenu_sentiment: Float?,                  // Score [-1.0, 1.0]
    contenu_intensite: Float?,                  // Intensité [0.0, 1.0]
    contenu_tags: List<String>,                 // Tags libres
    contenu_embeddings: List<Float>,            // Vecteur 384 dims (all-MiniLM-L6-v2)
    
    // Contexte
    contexte_domaine: String,                   // "sevrage_nicotine", "education", etc.
    contexte_phase: String?,                    // "J5", "semaine_3", etc.
    contexte_heure_journee: Int?,               // 0-23
    contexte_jour_semaine: Int?,                // 0-6 (0=lundi)
    
    // Métadonnées système
    created_at: DateTime,                       // Date création (auto)
    updated_at: DateTime?                       // Date dernière modif (optionnel)
})

// ============================================
// ARÊTES : TEMPOREL_SUITE
// ============================================

// Relation de succession chronologique
-[:TEMPOREL_SUITE {
    created_at: DateTime                        // Date création arête
}]->

// Sémantique : (a)-[:TEMPOREL_SUITE]->(b) signifie "a précède immédiatement b"
// Créées automatiquement par Greffier lors de l'ingestion

// ============================================
// ARÊTES : CORRELATION_OBSERVEE
// ============================================

// Relation de corrélation détectée
-[:CORRELATION_OBSERVEE {
    // Métriques
    force: Float,                               // Score confiance [0.0, 1.0]
    occurrences: Int,                           // Nombre fois pattern observé
    
    // Type pattern
    pattern_type: String,                       // Enum (voir 5.4)
    
    // Contexte détection
    detected_by: String,                        // "temporal_detector", "semantic_detector", etc.
    created_at: DateTime,                       // Date création
    last_updated: DateTime,                     // Dernière validation
    last_validated: DateTime?,                  // Dernière validation utilisateur (optionnel)
    
    // Statistiques
    counter_examples: Int?                      // Nombre contre-exemples (optionnel, Phase 1+)
}]->

// Sémantique : (a)-[:CORRELATION_OBSERVEE]->(b) signifie 
// "a et b sont corrélés (temporellement ou sémantiquement)"
// NOTE : Pas de direction causale, juste observation corrélation
```

### 5.3 Types d'Événements (Enum)

```python
# Type événements supportés Phase 0
EVENT_TYPES = {
    # Interactions conversationnelles
    "user_message": "Message utilisateur vers agent",
    "agent_response": "Réponse agent vers utilisateur",
    "tool_call": "Appel outil par agent (web_search, etc.)",
    
    # États observés
    "etat_emotionnel": "État émotionnel rapporté (joie, tristesse, anxiété, etc.)",
    "etat_physique": "État physique rapporté (fatigue, douleur, énergie, etc.)",
    "craving": "Pulsion/envie (contexte addiction)",
    "insomnie": "Trouble sommeil",
    "fatigue": "Fatigue physique/mentale",
    
    # Interventions
    "intervention": "Action thérapeutique délibérée",
    "external_event": "Événement externe significatif (piano, sport, etc.)",
    "meditation": "Session méditation/mindfulness",
    "exercice_physique": "Activité physique",
    
    # Résultats
    "resultat_mesure": "Outcome mesurable d'une intervention",
    "symptome": "Symptôme physique/mental observé",
    
    # Contexte
    "contexte_social": "Interaction sociale significative",
    "contexte_professionnel": "Événement lié au travail",
    "contexte_environnemental": "Changement environnement (météo, lieu, etc.)"
}
```

**Extensibilité :** Types additionnels ajoutables sans migration (schéma flexible)

### 5.4 Types de Patterns Corrélation (Enum)

```python
# Types patterns détectés par Juge V.E.
PATTERN_TYPES = {
    # Patterns temporels
    "temporal_negative_to_craving": "Sentiment négatif suivi de craving",
    "temporal_intervention_to_outcome": "Intervention suivie de résultat",
    "temporal_generic": "Co-occurrence temporelle générique",
    
    # Patterns sémantiques
    "semantic_similarity": "Similarité thématique (embeddings)",
    
    # Patterns séquentiels
    "sequential_insomnia_fatigue": "Insomnie → Fatigue",
    "sequential_craving_intervention": "Craving → Intervention",
    "sequential_generic": "Séquence A → B récurrente",
    
    # Patterns cycliques (Phase 1+)
    "cyclical_weekly": "Pattern hebdomadaire",
    "cyclical_daily": "Pattern quotidien"
}
```

### 5.5 Exemples de Requêtes Cypher

#### 5.5.1 Requêtes Basiques

```cypher
// Récupérer tous événements d'un utilisateur (chronologique)
MATCH (e:Event {user_id: $user_id})
RETURN e
ORDER BY e.timestamp_start ASC;

// Compter événements par type
MATCH (e:Event {user_id: $user_id})
RETURN e.type as type, count(*) as count
ORDER BY count DESC;

// Événements avec sentiment négatif
MATCH (e:Event {user_id: $user_id})
WHERE e.contenu_sentiment < -0.3
RETURN e.timestamp_start, e.contenu_texte, e.contenu_sentiment
ORDER BY e.timestamp_start DESC
LIMIT 20;
```

#### 5.5.2 Requêtes Temporelles

```cypher
// Événements dernières 24h
MATCH (e:Event {user_id: $user_id})
WHERE e.timestamp_start > datetime() - duration('P1D')
RETURN e
ORDER BY e.timestamp_start DESC;

// Événements entre deux dates
MATCH (e:Event {user_id: $user_id})
WHERE e.timestamp_start >= datetime($start_date)
  AND e.timestamp_start <= datetime($end_date)
RETURN e;

// Succession temporelle (A suivi de B dans 2h)
MATCH (a:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..10]->(b:Event)
WHERE duration.between(a.timestamp_start, b.timestamp_start) < duration('PT2H')
  AND a.type = 'craving'
  AND b.type = 'intervention'
RETURN a, b;
```

#### 5.5.3 Requêtes Corrélations

```cypher
// Toutes corrélations fortes
MATCH (a:Event {user_id: $user_id})-[r:CORRELATION_OBSERVEE]->(b:Event)
WHERE r.force > 0.7
RETURN a.contenu_texte, b.contenu_texte, r.force, r.occurrences
ORDER BY r.force DESC;

// Corrélations impliquant un type spécifique
MATCH (a:Event {user_id: $user_id})-[r:CORRELATION_OBSERVEE]->(b:Event)
WHERE (a.type = 'craving' OR b.type = 'craving')
  AND r.force > 0.6
RETURN a, r, b;

// Chemins de corrélations (A → B → C)
MATCH path = (a:Event {user_id: $user_id})-[:CORRELATION_OBSERVEE*2..3]->(c:Event)
WHERE all(r IN relationships(path) WHERE r.force > 0.6)
RETURN path
LIMIT 10;
```

#### 5.5.4 Requêtes Analytiques

```cypher
// Distribution sentiment par heure de journée
MATCH (e:Event {user_id: $user_id})
WHERE e.contenu_sentiment IS NOT NULL
RETURN e.contexte_heure_journee as heure,
       avg(e.contenu_sentiment) as sentiment_moyen,
       count(*) as nombre_evenements
ORDER BY heure;

// Efficacité interventions (sentiment avant vs après)
MATCH (avant:Event {user_id: $user_id})-[:TEMPOREL_SUITE*1..5]->(intervention:Event)-[:TEMPOREL_SUITE*1..5]->(apres:Event)
WHERE intervention.type = 'intervention'
  AND avant.contenu_sentiment IS NOT NULL
  AND apres.contenu_sentiment IS NOT NULL
  AND duration.between(avant.timestamp_start, intervention.timestamp_start) < duration('PT1H')
  AND duration.between(intervention.timestamp_start, apres.timestamp_start) < duration('PT2H')
RETURN intervention.contenu_texte as intervention,
       avg(avant.contenu_sentiment) as sentiment_avant,
       avg(apres.contenu_sentiment) as sentiment_apres,
       avg(apres.contenu_sentiment) - avg(avant.contenu_sentiment) as delta_sentiment,
       count(*) as occurrences
ORDER BY delta_sentiment DESC;

// Patterns récurrents par jour de semaine
MATCH (a:Event {user_id: $user_id})-[r:CORRELATION_OBSERVEE]->(b:Event)
RETURN a.contexte_jour_semaine as jour,
       a.type as type_a,
       b.type as type_b,
       count(*) as occurrences
ORDER BY jour, occurrences DESC;
```

### 5.6 Volumes de Données Estimés (Phase 0)

**Hypothèses :**
- 1 utilisateur (Matthias)
- 90 jours
- 50-100 événements/jour

**Estimations :**

```
Nœuds (Events) :
- Minimum : 50 * 90 = 4,500 nœuds
- Maximum : 100 * 90 = 9,000 nœuds
- Taille moyenne par nœud : ~2 KB (avec embeddings)
- Stockage total : 9 MB - 18 MB

Arêtes TEMPOREL_SUITE :
- ~1 arête par événement = 4,500 - 9,000 arêtes
- Taille moyenne : ~100 bytes
- Stockage : ~0.5 MB - 1 MB

Arêtes CORRELATION_OBSERVEE :
- Estimé : 50-200 patterns détectés sur 90 jours
- Avec échantillonnage (5 nœuds par pattern) : 250-1000 arêtes
- Stockage : ~25 KB - 100 KB

Total Stockage Phase 0 : ~10 MB - 20 MB
```

**Conclusion :** Volumétrie très faible Phase 0, optimisations performance non-nécessaires

### 5.7 Évolution Schéma (Phase 1+)

**Ajouts potentiels futurs :**

```cypher
// Nœuds additionnels
(:User {                                // Nœud utilisateur
    id: String,
    created_at: DateTime,
    metadata: Map
})

(:Session {                             // Nœud session
    id: String,
    start_time: DateTime,
    end_time: DateTime,
    summary: String
})

(:Pattern {                             // Nœud pattern agrégé
    id: String,
    type: String,
    description: String,
    confidence: Float,
    validated: Boolean
})

// Arêtes additionnelles
-[:BELONGS_TO]->                        // Event → User
-[:IN_SESSION]->                        // Event → Session
-[:VALIDATES]->                         // User → Pattern (validation explicite)
-[:CONTRADICTS]->                       // Event → Pattern (contre-exemple)
```

---

## 6. FLUX DE DONNÉES

### 6.1 Flux Ingestion Événement

```
┌─────────────┐
│ UTILISATEUR │
│ (Matthias)  │
└──────┬──────┘
       │
       │ 1. Interaction (message, événement)
       ▼
┌─────────────────────┐
│  CLIENT LOGGER      │
│  (EventLogger CLI)  │
└──────┬──────────────┘
       │
       │ 2. HTTP POST /log_event
       │    {agent, user_id, type, contenu, contexte}
       ▼
┌──────────────────────────────────────────┐
│         LE GREFFIER (API)                │
│                                          │
│  ┌────────────────────────────────────┐ │
│  │ 3. VALIDATION                      │ │
│  │    - Schéma Pydantic               │ │
│  │    - Sanitization XSS              │ │
│  │    - Vérif types/formats           │ │
│  └────────────┬───────────────────────┘ │
│               │                          │
│               ▼                          │
│  ┌────────────────────────────────────┐ │
│  │ 4. ENRICHISSEMENT                  │ │
│  │    - Sentiment Analysis (ML)       │ │
│  │      * Chargement modèle BERT      │ │
│  │      * Analyse texte               │ │
│  │      * Score [-1, 1]               │ │
│  │    - Embedding Generation          │ │
│  │      * SentenceTransformer         │ │
│  │      * Vecteur 384 dims            │ │
│  │    - Timestamps (UTC)              │ │
│  │    - UUID génération               │ │
│  └────────────┬───────────────────────┘ │
│               │                          │
│               ▼                          │
│  ┌────────────────────────────────────┐ │
│  │ 5. PERSISTANCE                     │ │
│  │    - Création nœud Neo4j           │ │
│  │    - Transaction atomique          │ │
│  │    - Retry logic (3 attempts)      │ │
│  └────────────┬───────────────────────┘ │
│               │                          │
└───────────────┼──────────────────────────┘
                │
                │ 6. Bolt Protocol
                ▼
┌────────────────────────────────────────┐
│         NEO4J DATABASE                 │
│                                        │
│  ┌──────────────────────────────────┐ │
│  │ WRITE :                          │ │
│  │                                  │ │
│  │ CREATE (e:Event {                │ │
│  │   id: $uuid,                     │ │
│  │   timestamp_start: $ts,          │ │
│  │   ...                            │ │
│  │   contenu_sentiment: $sentiment, │ │
│  │   contenu_embeddings: $embeddings│ │
│  │ })                               │ │
│  │                                  │ │
│  │ // Créer arête temporelle        │ │
│  │ MATCH (prev:Event)               │ │
│  │ WHERE prev.timestamp < e.ts      │ │
│  │ ORDER BY prev.timestamp DESC     │ │
│  │ LIMIT 1                          │ │
│  │ CREATE (prev)-[:TEMPOREL_SUITE]->│ │
│  │        (e)                        │ │
│  └──────────────────────────────────┘ │
│                                        │
└────────────────────────────────────────┘
                │
                │ 7. Response
                ▼
┌────────────────────────────────────────┐
│         LE GREFFIER (API)              │
│  Return:                               │
│  {                                     │
│    status: "success",                  │
│    node_id: "uuid",                    │
│    timestamp: "2025-10-26T..."         │
│  }                                     │
└────────────┬───────────────────────────┘
             │
             │ 8. HTTP 200 OK
             ▼
┌────────────────────┐
│  CLIENT LOGGER     │
│  ✓ Événement loggé │
└────────────────────┘
```

**Latence Typique :**
- Validation : <10ms
- Sentiment Analysis : 50-200ms (CPU)
- Embedding Generation : 20-100ms (CPU)
- Neo4j Write : 10-50ms
- **Total : 100-400ms**

**Gestion Erreurs :**
- Validation échoue → HTTP 422 (Unprocessable Entity)
- ML échoue → Fallback valeurs neutres (sentiment=0.0)
- Neo4j échoue → Retry 3x avec backoff exponentiel
- Échec final → HTTP 500, log erreur, alert monitoring

---

### 6.2 Flux Détection Corrélations (Batch)

```
┌────────────────────┐
│  CRON SCHEDULER    │
│  (3:00 AM daily)   │
└─────────┬──────────┘
          │
          │ 1. Trigger
          ▼
┌───────────────────────────────────────────┐
│      LE JUGE V.E. (Batch Service)         │
│                                           │
│  ┌─────────────────────────────────────┐ │
│  │ 2. INITIALIZATION                   │ │
│  │    - Connexion Neo4j                │ │
│  │    - Chargement config              │ │
│  │    - Liste users à analyser         │ │
│  └────────────┬────────────────────────┘ │
│               │                           │
│               ▼                           │
│  ┌─────────────────────────────────────┐ │
│  │ 3. POUR CHAQUE UTILISATEUR          │ │
│  │    user_id in user_ids:             │ │
│  └────────────┬────────────────────────┘ │
│               │                           │
│               ▼                           │
│  ┌─────────────────────────────────────┐ │
│  │ 4. DÉTECTION TEMPORELLE             │ │
│  │    TemporalCooccurrenceDetector     │ │
│  │                                     │ │
│  │    a) Patterns sentiment → craving  │ │
│  │       - Query événements window 2h  │ │
│  │       - Count occurrences           │ │
│  │       - Si count >= threshold (3)   │ │
│  │         → CREATE arête CORRELATION  │ │
│  │                                     │ │
│  │    b) Patterns intervention → outcome│ │
│  │       - Query interventions         │ │
│  │       - Mesure sentiment après      │ │
│  │       - Calcul delta sentiment      │ │
│  │       - Si significatif             │ │
│  │         → CREATE arête CORRELATION  │ │
│  │                                     │ │
│  │    c) Patterns génériques A → B     │ │
│  │       - Query toutes séquences      │ │
│  │       - Count par (type_a, type_b)  │ │
│  │       - Top 10 patterns             │ │
│  │         → CREATE arêtes CORRELATION │ │
│  └────────────┬────────────────────────┘ │
│               │                           │
│               ▼                           │
│  ┌─────────────────────────────────────┐ │
│  │ 5. DÉTECTION SÉMANTIQUE             │ │
│  │    SemanticSimilarityDetector       │ │
│  │                                     │ │
│  │    a) Récupération événements       │ │
│  │       - Query events avec embeddings│ │
│  │       - Lookback 21 jours           │ │
│  │                                     │ │
│  │    b) Calcul matrice similarité     │ │
│  │       - Cosine similarity (sklearn) │ │
│  │       - Matrice n×n                 │ │
│  │                                     │ │
│  │    c) Création arêtes               │ │
│  │       - Pour chaque paire similarity│ │
│  │         >= threshold (0.75)         │ │
│  │         → CREATE arête CORRELATION  │ │
│  └────────────┬────────────────────────┘ │
│               │                           │
│               ▼                           │
│  ┌─────────────────────────────────────┐ │
│  │ 6. DÉTECTION SÉQUENTIELLE           │ │
│  │    SequentialPatternDetector        │ │
│  │                                     │ │
│  │    a) Patterns spécifiques          │ │
│  │       - Insomnie → Fatigue          │ │
│  │       - Craving → Intervention      │ │
│  │                                     │ │
│  │    b) Patterns génériques           │ │
│  │       - Séquences A → B récurrentes │ │
│  │       - Max delay 12h               │ │
│  │       - Calcul délai moyen          │ │
│  │                                     │ │
│  │    c) Création arêtes               │ │
│  │       - Si occurrences >= threshold │ │
│  │         → CREATE arête CORRELATION  │ │
│  └────────────┬────────────────────────┘ │
│               │                           │
│               ▼                           │
│  ┌─────────────────────────────────────┐ │
│  │ 7. LOGGING & STATS                  │ │
│  │    - Patterns détectés par type     │ │
│  │    - Arêtes créées                  │ │
│  │    - Durée exécution                │ │
│  │    - Erreurs éventuelles            │ │
│  └─────────────────────────────────────┘ │
│                                           │
└───────────────────────────────────────────┘
          │
          │ 8. Write to Neo4j
          ▼
┌────────────────────────────────────────┐
│         NEO4J DATABASE                 │
│                                        │
│  Nouvelles arêtes CORRELATION_OBSERVEE │
│  créées avec propriétés :              │
│    - force                             │
│    - occurrences                       │
│    - pattern_type                      │
│    - created_at                        │
└────────────────────────────────────────┘
```

**Durée Typique (1 user, 5000 events) :**
- Détection temporelle : 2-5 min
- Détection sémantique : 5-10 min (calculs intensifs)
- Détection séquentielle : 2-5 min
- **Total : 10-20 minutes**

**Optimisations Possibles (Phase 1+) :**
- Calcul incrémental (seulement nouveaux événements)
- Parallélisation détecteurs
- Cache résultats intermédiaires
- GPU pour calculs embeddings/similarité

---

### 6.3 Flux Enrichissement Contexte Conversationnel

```
┌─────────────┐
│ UTILISATEUR │
│ (Matthias)  │
└──────┬──────┘
       │
       │ 1. Message vers Claude
       │    "Je me sens fatigué aujourd'hui"
       ▼
┌──────────────────────────────────────┐
│    INTEGRATION LAYER                 │
│    (Script/App intermédiaire)        │
└──────┬───────────────────────────────┘
       │
       │ 2. Log événement + Récupération contexte
       │
       ├─────────────────────┐
       │                     │
       ▼                     ▼
┌──────────────┐   ┌─────────────────────────────┐
│  GREFFIER    │   │  MÉMORIALISTE V.E.          │
│              │   │                             │
│  Log message │   │  enrich_context(            │
│  utilisateur │   │    user_id,                 │
└──────────────┘   │    message,                 │
                   │    session_id               │
                   │  )                          │
                   └────────┬────────────────────┘
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
       ▼                    ▼                    ▼
┌──────────────┐  ┌─────────────────┐  ┌────────────────┐
│ TEMPORAL     │  │ SEMANTIC        │  │ CORRELATION    │
│ RETRIEVER    │  │ RETRIEVER       │  │ RETRIEVER      │
│              │  │                 │  │                │
│ Query:       │  │ 1. Encode msg   │  │ Query:         │
│ - Last 5     │  │    (embedding)  │  │ - Correlations │
│   events     │  │ 2. Similarity   │  │   pertinentes  │
│ - Today      │  │    search       │  │ - Force > 0.6  │
│   events     │  │ 3. Top-3 similar│  │ - Contextuelles│
└──────┬───────┘  └────────┬────────┘  └────────┬───────┘
       │                   │                    │
       │                   │                    │
       └────────────┬──────┴──────┬─────────────┘
                    │             │
                    ▼             │
       ┌──────────────────────────┴──────────┐
       │  NEO4J DATABASE                     │
       │                                     │
       │  Cypher Queries:                    │
       │  - MATCH recent events              │
       │  - Cosine similarity embeddings     │
       │  - MATCH correlation edges          │
       └────────────────┬────────────────────┘
                        │
                        │ Results
                        ▼
       ┌────────────────────────────────────┐
       │  MÉMORIALISTE V.E.                 │
       │                                    │
       │  Résultats récupérés :             │
       │  - temporal_events: [...]          │
       │  - semantic_events: [...]          │
       │  - correlations: [...]             │
       │                                    │
       │  ▼ Format Prompt                   │
       │                                    │
       │  EmpatheticPromptFormatter         │
       │  .format_enriched_prompt(...)      │
       └────────────────┬───────────────────┘
                        │
                        │ 3. Prompt Enrichi
                        ▼
       ┌────────────────────────────────────┐
       │  PROMPT ENRICHI                    │
       │                                    │
       │  === CONTEXTE MÉMORIEL ===         │
       │                                    │
       │  MESSAGE ACTUEL:                   │
       │  "Je me sens fatigué aujourd'hui"  │
       │  Sentiment: -0.5                   │
       │                                    │
       │  CONTINUITÉ TEMPORELLE:            │
       │  • Il y a 8h: "Mal dormi cette nuit│
       │    [sentiment: -0.6]               │
       │  • Il y a 2j: "Épuisé après réunion│
       │    [sentiment: -0.7]               │
       │                                    │
       │  CONNEXIONS SÉMANTIQUES:           │
       │  • Il y a 1 semaine (sim: 0.82):   │
       │    "Vraiment crevé ce matin"       │
       │                                    │
       │  PATTERNS OBSERVÉS:                │
       │  • Pattern: "insomnie" souvent     │
       │    suivi de "fatigue" (5 fois,     │
       │    force: 0.85)                    │
       │                                    │
       │  === INSTRUCTIONS ===              │
       │  1. Utilise ces souvenirs...       │
       │  2. Ne conclus pas causalité...    │
       │  3. Présente comme questions...    │
       │  ...                               │
       └────────────────┬───────────────────┘
                        │
                        │ 4. Envoi à LLM
                        ▼
       ┌────────────────────────────────────┐
       │  ANTHROPIC API (Claude)            │
       │                                    │
       │  messages.create(                  │
       │    model="claude-sonnet-4.5",      │
       │    messages=[{                     │
       │      role: "user",                 │
       │      content: enriched_prompt      │
       │    }]                              │
       │  )                                 │
       └────────────────┬───────────────────┘
                        │
                        │ 5. Réponse Claude
                        ▼
       ┌────────────────────────────────────┐
       │  RÉPONSE EMPATHIQUE                │
       │                                    │
       │  "J'ai remarqué quelque chose :    │
       │  tu m'as dit cette nuit que tu     │
       │  avais mal dormi, et maintenant tu │
       │  te sens fatigué. C'est quelque    │
       │  chose que j'ai vu se répéter      │
       │  plusieurs fois - quand tu as du   │
       │  mal à dormir, tu sembles épuisé   │
       │  le lendemain. Comment s'est passée│
       │  ta nuit exactement ? Et comment   │
       │  tu te sens là, maintenant ?"      │
       └────────────────┬───────────────────┘
                        │
                        │ 6. Retour utilisateur
                        ▼
       ┌────────────────────────────────────┐
       │  GREFFIER                          │
       │  Log réponse agent                 │
       └────────────────────────────────────┘
                        │
                        ▼
       ┌────────────────────────────────────┐
       │  NEO4J DATABASE                    │
       │  Nouvel Event (agent_response)     │
       └────────────────────────────────────┘
```

**Latence Typique :**
- Retrieval (temporal + semantic + correlation) : 200-500ms
- Formatting prompt : <50ms
- Anthropic API call : 2-5 secondes (génération LLM)
- **Total perçu : 2-6 secondes**

**Optimisations Possibles :**
- Cache retrieval results (Redis) si message similaire récent
- Parallel retrieval (temporal + semantic + correlation)
- Streaming response (Anthropic API)

---

## 7. INTERFACES & APIs

### 7.1 API Greffier (REST)

#### 7.1.1 Spécification OpenAPI

```yaml
openapi: 3.0.3
info:
  title: Greffier API
  description: Event Ingestion Service - Système MTE
  version: 1.0.0
  contact:
    name: Rodin (Claude)

servers:
  - url: http://localhost:8000
    description: Local development
  - url: https://greffier.mte.example.com
    description: Production (Phase 1+)

paths:
  /health:
    get:
      summary: Health check
      operationId: healthCheck
      responses:
        '200':
          description: Service healthy
          content:
            application/json:
              schema:
                type: object
                properties:
                  status:
                    type: string
                    example: "healthy"
                  service:
                    type: string
                    example: "greffier"
                  version:
                    type: string
                    example: "1.0.0"

  /log_event:
    post:
      summary: Log un événement
      operationId: logEvent
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/EventCreate'
      responses:
        '200':
          description: Événement loggé avec succès
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/LogResponse'
        '422':
          description: Validation error
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ValidationError'
        '500':
          description: Erreur serveur
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/ErrorResponse'

  /stats:
    get:
      summary: Statistiques système
      operationId: getStats
      parameters:
        - name: user_id
          in: query
          required: false
          schema:
            type: string
      responses:
        '200':
          description: Statistiques
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/StatsResponse'

components:
  schemas:
    EventCreate:
      type: object
      required:
        - agent
        - user_id
        - session_id
        - type
        - contenu
        - contexte
      properties:
        agent:
          type: string
          example: "rodin"
        user_id:
          type: string
          example: "user_abc123_hash"
        session_id:
          type: string
          example: "session_xyz789"
        type:
          type: string
          enum:
            - user_message
            - agent_response
            - craving
            - intervention
            - external_event
            - etat_emotionnel
            - etat_physique
          example: "user_message"
        contenu:
          $ref: '#/components/schemas/EventContenu'
        contexte:
          $ref: '#/components/schemas/EventContexte'

    EventContenu:
      type: object
      required:
        - texte
      properties:
        texte:
          type: string
          minLength: 1
          maxLength: 10000
          example: "Je me sens vraiment fatigué aujourd'hui"
        sentiment_detecte:
          type: number
          format: float
          minimum: -1.0
          maximum: 1.0
          nullable: true
        intensite:
          type: number
          format: float
          minimum: 0.0
          maximum: 1.0
          nullable: true
        tags:
          type: array
          items:
            type: string
          example: ["fatigue", "emotionnel"]

    EventContexte:
      type: object
      required:
        - domaine
      properties:
        domaine:
          type: string
          example: "sevrage_nicotine"
        phase:
          type: string
          nullable: true
          example: "J5"
        heure_journee:
          type: integer
          minimum: 0
          maximum: 23
          nullable: true
        jour_semaine:
          type: integer
          minimum: 0
          maximum: 6
          nullable: true
        metadata:
          type: object
          additionalProperties: true

    LogResponse:
      type: object
      properties:
        status:
          type: string
          example: "success"
        node_id:
          type: string
          format: uuid
          example: "550e8400-e29b-41d4-a716-446655440000"
        timestamp:
          type: string
          format: date-time
          example: "2025-10-26T14:30:00Z"
        message:
          type: string
          nullable: true
          example: "Événement loggé avec succès"

    StatsResponse:
      type: object
      properties:
        total_events:
          type: integer
          example: 4523
        total_edges:
          type: integer
          example: 4612
        correlations_count:
          type: integer
          example: 87
        date_range:
          type: object
          properties:
            start:
              type: string
              format: date-time
            end:
              type: string
              format: date-time

    ValidationError:
      type: object
      properties:
        detail:
          type: array
          items:
            type: object
            properties:
              loc:
                type: array
                items:
                  type: string
              msg:
                type: string
              type:
                type: string

    ErrorResponse:
      type: object
      properties:
        detail:
          type: string
          example: "Internal server error"
```

#### 7.1.2 Exemples d'Appels

**cURL - Log User Message**

```bash
curl -X POST http://localhost:8000/log_event \
  -H "Content-Type: application/json" \
  -d '{
    "agent": "rodin",
    "user_id": "matthias_hash_abc123",
    "session_id": "session_sevrage_001",
    "type": "user_message",
    "contenu": {
      "texte": "Je me sens vraiment fatigué aujourd'\''hui",
      "tags": ["fatigue", "emotionnel"]
    },
    "contexte": {
      "domaine": "sevrage_nicotine",
      "phase": "J5",
      "heure_journee": 14,
      "jour_semaine": 4
    }
  }'
```

**Python - Log External Event**

```python
import requests

event = {
    "agent": "rodin",
    "user_id": "matthias_hash_abc123",
    "session_id": "session_sevrage_001",
    "type": "external_event",
    "contenu": {
        "texte": "40min piano - One Summer's Day",
        "tags": ["piano", "intervention", "flow"]
    },
    "contexte": {
        "domaine": "sevrage_nicotine",
        "phase": "J5",
        "heure_journee": 20
    }
}

response = requests.post(
    "http://localhost:8000/log_event",
    json=event
)

print(response.json())
# Output: {"status": "success", "node_id": "...", "timestamp": "..."}
```

### 7.2 Client Logger (CLI)

```python
# event_logger.py - Client CLI complet

import requests
import json
from datetime import datetime
from typing import Optional, List, Dict
import argparse
import sys

class EventLogger:
    """Client CLI pour logging événements"""
    
    def __init__(self, api_url: str, user_id: str, session_id: str):
        self.api_url = api_url
        self.user_id = user_id
        self.session_id = session_id
    
    def log(
        self,
        type: str,
        texte: str,
        sentiment: Optional[float] = None,
        intensite: Optional[float] = None,
        tags: Optional[List[str]] = None,
        domaine: str = "sevrage_nicotine",
        phase: Optional[str] = None
    ) -> Dict:
        """
        Log un événement
        
        Returns:
            Dict: Response API
        """
        # Contexte auto
        now = datetime.now()
        
        event = {
            "agent": "rodin",
            "user_id": self.user_id,
            "session_id": self.session_id,
            "type": type,
            "contenu": {
                "texte": texte,
                "sentiment_detecte": sentiment,
                "intensite": intensite,
                "tags": tags or []
            },
            "contexte": {
                "domaine": domaine,
                "phase": phase,
                "heure_journee": now.hour,
                "jour_semaine": now.weekday()
            }
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/log_event",
                json=event,
                timeout=10
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Erreur logging: {e}", file=sys.stderr)
            return {"status": "error", "message": str(e)}
    
    # Helpers
    def message(self, texte: str, sentiment: float = None):
        """Log message utilisateur"""
        return self.log("user_message", texte, sentiment=sentiment)
    
    def craving(self, intensite: float, description: str = ""):
        """Log craving"""
        texte = f"Craving - intensité {intensite}/10"
        if description:
            texte += f": {description}"
        return self.log("craving", texte, intensite=intensite/10)
    
    def intervention(self, texte: str, tags: List[str] = None):
        """Log intervention"""
        return self.log("intervention", texte, tags=tags)
    
    def event(self, texte: str, tags: List[str] = None):
        """Log événement externe"""
        return self.log("external_event", texte, tags=tags)
    
    def etat(self, type_etat: str, texte: str, sentiment: float = None):
        """Log état (émotionnel ou physique)"""
        return self.log(type_etat, texte, sentiment=sentiment)

def interactive_mode(logger: EventLogger):
    """Mode interactif CLI"""
    print("=== Event Logger - Mode Interactif ===")
    print("Commandes: message, craving, intervention, event, etat, quit")
    print()
    
    while True:
        try:
            cmd = input("\n📝 Commande: ").strip().lower()
            
            if cmd == "quit" or cmd == "exit":
                print("Bye! 👋")
                break
            
            elif cmd == "message":
                texte = input("Texte: ")
                sentiment_input = input("Sentiment [-1 to 1] (optionnel): ")
                sentiment = float(sentiment_input) if sentiment_input else None
                result = logger.message(texte, sentiment)
                print(f"✅ Loggé: {result.get('node_id', 'N/A')}")
            
            elif cmd == "craving":
                intensite = float(input("Intensité [0-10]: "))
                desc = input("Description (optionnel): ")
                result = logger.craving(intensite, desc)
                print(f"✅ Loggé: {result.get('node_id', 'N/A')}")
            
            elif cmd == "intervention":
                texte = input("Description: ")
                tags_input = input("Tags (comma-separated): ")
                tags = [t.strip() for t in tags_input.split(",")] if tags_input else None
                result = logger.intervention(texte, tags)
                print(f"✅ Loggé: {result.get('node_id', 'N/A')}")
            
            elif cmd == "event":
                texte = input("Description: ")
                tags_input = input("Tags (comma-separated): ")
                tags = [t.strip() for t in tags_input.split(",")] if tags_input else None
                result = logger.event(texte, tags)
                print(f"✅ Loggé: {result.get('node_id', 'N/A')}")
            
            elif cmd == "etat":
                type_etat = input("Type (etat_emotionnel/etat_physique): ")
                texte = input("Description: ")
                sentiment_input = input("Sentiment [-1 to 1] (optionnel): ")
                sentiment = float(sentiment_input) if sentiment_input else None
                result = logger.etat(type_etat, texte, sentiment)
                print(f"✅ Loggé: {result.get('node_id', 'N/A')}")
            
            else:
                print("❌ Commande inconnue")
        
        except KeyboardInterrupt:
            print("\n\nBye! 👋")
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

def main():
    parser = argparse.ArgumentParser(description="Event Logger CLI")
    parser.add_argument("--api-url", default="http://localhost:8000", help="API URL")
    parser.add_argument("--user-id", required=True, help="User ID")
    parser.add_argument("--session-id", default=None, help="Session ID")
    parser.add_argument("--interactive", "-i", action="store_true", help="Mode interactif")
    
    # Arguments pour mode non-interactif
    parser.add_argument("--type", help="Event type")
    parser.add_argument("--text", help="Event text")
    parser.add_argument("--tags", help="Tags (comma-separated)")
    
    args = parser.parse_args()
    
    # Session ID auto si non fourni
    session_id = args.session_id or f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    logger = EventLogger(args.api_url, args.user_id, session_id)
    
    if args.interactive:
        interactive_mode(logger)
    else:
        # Mode one-shot
        if not args.type or not args.text:
            print("❌ --type et --text requis en mode non-interactif")
            sys.exit(1)
        
        tags = [t.strip() for t in args.tags.split(",")] if args.tags else None
        result = logger.log(args.type, args.text, tags=tags)
        print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
```

**Usage:**

```bash
# Mode interactif
python event_logger.py --user-id matthias_hash --interactive

# Mode one-shot
python event_logger.py \
  --user-id matthias_hash \
  --type user_message \
  --text "Je me sens mieux aujourd'hui" \
  --tags "emotionnel,positif"
```

---

## 8. SÉCURITÉ & CONFIDENTIALITÉ

### 8.1 Principes de Sécurité

**1. Defense in Depth**
- Validation à chaque couche (client, API, database)
- Sanitization inputs
- Least privilege (permissions minimales)

**2. Privacy by Design**
- Anonymisation user_id (hash irréversible)
- Pas de PII (Personally Identifiable Information) en clair
- Chiffrement données sensibles

**3. Data Minimization**
- Collecte uniquement données nécessaires
- Pas de logging excessif
- Retention policies (Phase 1+)

### 8.2 Authentification & Autorisation

**Phase 0 (Développement) :**
- Pas d'authentification (single user local)
- API accessible localhost uniquement

**Phase 1+ (Production) :**

```python
# API Key authentication
from fastapi import Security, HTTPException
from fastapi.security.api_key import APIKeyHeader

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != settings.api_key:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

# Protected endpoint
@app.post("/log_event")
async def log_event(
    event: EventCreate,
    api_key: str = Depends(get_api_key)
):
    # ...
```

**JWT pour multi-users (Phase 2+) :**

```python
from fastapi import Depends
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret,
            algorithms=["HS256"]
        )
        user_id = payload.get("sub")
        if user_id is None:
            raise HTTPException(status_code=401)
        return user_id
    except JWTError:
        raise HTTPException(status_code=401)
```

### 8.3 Chiffrement

**Données en Transit :**
```nginx
# HTTPS obligatoire (Nginx config Phase 1+)
server {
    listen 443 ssl http2;
    server_name greffier.mte.example.com;
    
    ssl_certificate /etc/letsencrypt/live/domain/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/domain/privkey.pem;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    
    location / {
        proxy_pass http://localhost:8000;
    }
}
```

**Données au Repos (Neo4j) :**

```bash
# Neo4j encryption at rest (Enterprise Edition)
dbms.directories.data=/var/lib/neo4j/data
dbms.security.encryption_provider=lucene-native
dbms.security.encryption.keystore_path=/path/to/keystore
dbms.security.encryption.keystore_password=<password>
```

**Embeddings Sensibles :**
- Embeddings ne sont pas chiffrés (computationnellement coûteux)
- Considérés "derived data" (pas PII direct)
- Si nécessaire Phase 1+ : chiffrement sélectif colonnes Neo4j

### 8.4 Anonymisation

**User ID Hashing :**

```python
import hashlib
import os

def hash_user_id(email: str, salt: str = None) -> str:
    """
    Hash user identifier de manière irréversible
    
    Args:
        email: Email ou identifier utilisateur
        salt: Salt optionnel (stocké séparément)
        
    Returns:
        str: Hash SHA-256
    """
    if salt is None:
        salt = os.environ.get("USER_ID_SALT", "default_salt_change_me")
    
    data = f"{email}{salt}".encode('utf-8')
    return hashlib.sha256(data).hexdigest()

# Usage
user_id_hash = hash_user_id("matthias@example.com")
# Output: "5f3d8e7c2b1a9e4f..."
```

**Pseudonymisation Textes (Optionnel Phase 1+) :**

```python
import re
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine

analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()

def anonymize_text(text: str) -> str:
    """
    Pseudonymise PII dans texte
    
    Détecte et remplace:
    - Noms
    - Emails
    - Numéros téléphone
    - Adresses
    """
    results = analyzer.analyze(
        text=text,
        language='fr',
        entities=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER", "LOCATION"]
    )
    
    anonymized = anonymizer.anonymize(
        text=text,
        analyzer_results=results
    )
    
    return anonymized.text

# Exemple
text = "Je suis Matthias, j'habite à Paris"
anonymized = anonymize_text(text)
# Output: "Je suis <PERSON>, j'habite à <LOCATION>"
```

### 8.5 Validation & Sanitization

**XSS Prevention :**

```python
import html
import re

def sanitize_html(text: str) -> str:
    """Remove HTML tags et escape caractères spéciaux"""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Escape HTML entities
    text = html.escape(text)
    return text

def sanitize_sql(text: str) -> str:
    """Prevent SQL injection (bien que Neo4j parameterized queries)"""
    # Cypher parameterized queries sont safe, mais précaution
    dangerous_patterns = [
        r'\bDROP\b', r'\bDELETE\b', r'\bINSERT\b',
        r'\bUPDATE\b', r'\bCREATE\b', r'\bMERGE\b'
    ]
    for pattern in dangerous_patterns:
        if re.search(pattern, text, re.IGNORECASE):
            raise ValueError("Potentially dangerous input detected")
    return text
```

**Input Validation (Pydantic) :**

```python
from pydantic import BaseModel, validator, Field

class EventContenu(BaseModel):
    texte: str = Field(..., min_length=1, max_length=10000)
    
    @validator('texte')
    def sanitize_texte(cls, v):
        # Remove null bytes
        v = v.replace('\x00', '')
        # Remove excessive whitespace
        v = re.sub(r'\s+', ' ', v).strip()
        # Sanitize HTML
        v = sanitize_html(v)
        return v
    
    @validator('sentiment_detecte')
    def validate_sentiment(cls, v):
        if v is not None and (v < -1.0 or v > 1.0):
            raise ValueError("Sentiment doit être dans [-1.0, 1.0]")
        return v
```

### 8.6 Rate Limiting (Phase 1+)

```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/log_event")
@limiter.limit("100/minute")  # Max 100 requêtes/minute
async def log_event(request: Request, event: EventCreate):
    # ...
```

### 8.7 Audit Logging

```python
import logging
import json
from datetime import datetime

audit_logger = logging.getLogger("audit")

def log_audit_event(
    action: str,
    user_id: str,
    resource: str,
    result: str,
    metadata: dict = None
):
    """
    Log événement audit
    
    Args:
        action: Action effectuée (CREATE, READ, UPDATE, DELETE)
        user_id: Utilisateur (hash)
        resource: Ressource affectée
        result: Résultat (SUCCESS, FAILURE)
        metadata: Données additionnelles
    """
    audit_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "action": action,
        "user_id": user_id,
        "resource": resource,
        "result": result,
        "metadata": metadata or {}
    }
    
    audit_logger.info(json.dumps(audit_entry))

# Usage
log_audit_event(
    action="CREATE",
    user_id="matthias_hash",
    resource="Event:550e8400-e29b-41d4-a716-446655440000",
    result="SUCCESS",
    metadata={"type": "user_message", "ip": "127.0.0.1"}
)
```

### 8.8 Conformité RGPD

**Droits Utilisateurs :**

```python
# Droit d'accès (Article 15)
@app.get("/user/{user_id}/data")
async def export_user_data(user_id: str):
    """Export toutes données utilisateur (format JSON)"""
    with driver.session() as session:
        result = session.run("""
            MATCH (e:Event {user_id: $user_id})
            RETURN e
        """, user_id=user_id)
        events = [dict(record['e']) for record in result]
    return {"user_id": user_id, "events": events}

# Droit à l'oubli (Article 17)
@app.delete("/user/{user_id}/data")
async def delete_user_data(user_id: str):
    """Supprime toutes données utilisateur"""
    with driver.session() as session:
        session.run("""
            MATCH (e:Event {user_id: $user_id})
            DETACH DELETE e
        """, user_id=user_id)
    return {"status": "deleted", "user_id": user_id}

# Droit de rectification (Article 16)
@app.patch("/event/{event_id}")
async def update_event(event_id: str, updates: dict):
    """Modifie un événement spécifique"""
    # Implémenter logique update
    pass
```

**Consentement :**

```python
# Modèle consentement
class UserConsent(BaseModel):
    user_id: str
    data_collection: bool  # Consent collecte données
    data_analysis: bool    # Consent analyse patterns
    data_retention_days: int = 365  # Durée conservation
    consented_at: datetime
    
# Vérification avant logging
def check_consent(user_id: str) -> bool:
    # Query consent database
    consent = get_user_consent(user_id)
    return consent.data_collection if consent else False
```

---

## 9. PERFORMANCE & SCALABILITÉ

### 9.1 Optimisations Phase 0

**Neo4j Indexes (Déjà spécifiés Section 5.2) :**
- `event_id` (UNIQUE)
- `timestamp_start`, `timestamp_end`
- `user_id`, `session_id`
- `type`, `sentiment`

**Query Optimization :**

```cypher
// ❌ MAUVAIS : Full scan
MATCH (e:Event)
WHERE e.user_id = 'matthias_hash'
  AND e.contenu_sentiment < -0.3
RETURN e;

// ✅ BON : Utilise index
MATCH (e:Event {user_id: 'matthias_hash'})
WHERE e.contenu_sentiment < -0.3
RETURN e;

// ✅ EXCELLENT : Index + limite
MATCH (e:Event {user_id: 'matthias_hash'})
WHERE e.contenu_sentiment < -0.3
RETURN e
ORDER BY e.timestamp_start DESC
LIMIT 20;
```

**Caching (Redis - Phase 1+) :**

```python
import redis
import json
from functools import wraps

redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)

def cache_result(ttl_seconds: int = 300):
    """Decorator pour cache résultats"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Générer cache key
            cache_key = f"{func.__name__}:{json.dumps(args)}:{json.dumps(kwargs)}"
            
            # Check cache
            cached = redis_client.get(cache_key)
            if cached:
                return json.loads(cached)
            
            # Execute function
            result = await func(*args, **kwargs)
            
            # Store in cache
            redis_client.setex(
                cache_key,
                ttl_seconds,
                json.dumps(result)
            )
            
            return result
        return wrapper
    return decorator

# Usage
@cache_result(ttl_seconds=300)
async def get_recent_events(user_id: str, limit: int):
    # Query database
    pass
```

### 9.2 Scalabilité Horizontale (Phase 2+)

**Architecture Multi-Instances :**

```
┌──────────────────────────────────────────┐
│          LOAD BALANCER (Nginx)           │
└────────┬─────────────┬───────────────────┘
         │             │
    ┌────▼────┐   ┌────▼────┐
    │ API #1  │   │ API #2  │   ... (N instances)
    └────┬────┘   └────┬────┘
         │             │
         └──────┬──────┘
                │
    ┌───────────▼───────────┐
    │   Neo4j Cluster       │
    │   (Causal Cluster)    │
    │                       │
    │  ┌────┐ ┌────┐ ┌────┐│
    │  │Core│ │Core│ │Core││
    │  └────┘ └────┘ └────┘│
    │  ┌──────────────────┐ │
    │  │  Read Replicas   │ │
    │  └──────────────────┘ │
    └───────────────────────┘
```

**Neo4j Causal Cluster Config :**

```conf
# Core server
dbms.mode=CORE
causal_clustering.minimum_core_cluster_size_at_formation=3
causal_clustering.minimum_core_cluster_size_at_runtime=3
causal_clustering.initial_discovery_members=core1:5000,core2:5000,core3:5000

# Read replica
dbms.mode=READ_REPLICA
causal_clustering.initial_discovery_members=core1:5000,core2:5000,core3:5000
```

**Load Balancing Strategy :**
- Writes → Core servers (automatic leader election)
- Reads → Read replicas (distribué round-robin)

### 9.3 Métriques Performance

**Targets Phase 0 :**
- API latency p50 : < 200ms
- API latency p95 : < 500ms
- API latency p99 : < 1000ms
- Neo4j query time p50 : < 50ms
- Neo4j query time p95 : < 200ms
- Batch job duration : < 30min (pour 10k events)

**Monitoring (Prometheus + Grafana) :**

```python
from prometheus_client import Counter, Histogram, Gauge
import time

# Métriques
request_count = Counter(
    'api_requests_total',
    'Total API requests',
    ['method', 'endpoint', 'status']
)

request_duration = Histogram(
    'api_request_duration_seconds',
    'API request duration',
    ['method', 'endpoint']
)

event_count = Gauge(
    'total_events',
    'Total events in database'
)

# Instrumentation
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    duration = time.time() - start_time
    
    request_count.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    request_duration.labels(
        method=request.method,
        endpoint=request.url.path
    ).observe(duration)
    
    return response
```

---

## 10. DÉPLOIEMENT

### 10.1 Docker Compose (Phase 0)

```yaml
# docker-compose.yml
version: '3.8'

services:
  neo4j:
    image: neo4j:5.14.0
    container_name: mte-neo4j
    ports:
      - "7474:7474"  # HTTP
      - "7687:7687"  # Bolt
    environment:
      NEO4J_AUTH: neo4j/securepassword
      NEO4J_dbms_memory_pagecache_size: 1G
      NEO4J_dbms_memory_heap_initial__size: 1G
      NEO4J_dbms_memory_heap_max__size: 2G
    volumes:
      - ./data/neo4j:/data
      - ./logs/neo4j:/logs
      - ./backups/neo4j:/backups
    networks:
      - mte-network
    restart: unless-stopped

  greffier:
    build:
      context: ./greffier
      dockerfile: Dockerfile
    container_name: mte-greffier
    ports:
      - "8000:8000"
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: securepassword
      LOG_LEVEL: INFO
    volumes:
      - ./logs/greffier:/app/logs
    depends_on:
      - neo4j
    networks:
      - mte-network
    restart: unless-stopped

  juge:
    build:
      context: ./juge
      dockerfile: Dockerfile
    container_name: mte-juge
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: securepassword
      BATCH_SCHEDULE_TIME: "03:00"
      RUN_ON_STARTUP: "false"
    volumes:
      - ./logs/juge:/app/logs
    depends_on:
      - neo4j
    networks:
      - mte-network
    restart: unless-stopped

  memorialiste:
    build:
      context: ./memorialiste
      dockerfile: Dockerfile
    container_name: mte-memorialiste
    environment:
      NEO4J_URI: bolt://neo4j:7687
      NEO4J_USER: neo4j
      NEO4J_PASSWORD: securepassword
    volumes:
      - ./logs/memorialiste:/app/logs
    depends_on:
      - neo4j
    networks:
      - mte-network
    restart: unless-stopped

networks:
  mte-network:
    driver: bridge

volumes:
  neo4j-data:
  neo4j-logs:
  neo4j-backups:
```

**Démarrage :**

```bash
# Build et démarrer
docker-compose up -d

# Vérifier statuts
docker-compose ps

# Logs
docker-compose logs -f greffier

# Arrêter
docker-compose down

# Arrêter et supprimer volumes (ATTENTION: perte données)
docker-compose down -v
```

### 10.2 Dockerfile Greffier

```dockerfile
# greffier/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copier requirements
COPY requirements.txt .

# Installer dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Télécharger modèles ML (cache)
RUN python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')"
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Copier code application
COPY . .

# Créer répertoires logs
RUN mkdir -p /app/logs

# Exposer port
EXPOSE 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Commande démarrage
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 10.3 Scripts Utilitaires

**Backup Neo4j :**

```bash
#!/bin/bash
# scripts/backup_neo4j.sh

BACKUP_DIR="/backups/neo4j"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="neo4j_backup_${DATE}"

# Créer backup
docker exec mte-neo4j neo4j-admin database dump neo4j \
    --to-path=/backups/${BACKUP_NAME}.dump

# Compresser
gzip ${BACKUP_DIR}/${BACKUP_NAME}.dump

# Cleanup anciens backups (garder 7 derniers jours)
find ${BACKUP_DIR} -name "*.dump.gz" -mtime +7 -delete

echo "Backup créé: ${BACKUP_NAME}.dump.gz"
```

**Restore Neo4j :**

```bash
#!/bin/bash
# scripts/restore_neo4j.sh

if [ -z "$1" ]; then
    echo "Usage: ./restore_neo4j.sh <backup_file>"
    exit 1
fi

BACKUP_FILE=$1

# Arrêter Neo4j
docker-compose stop neo4j

# Restore
docker run --rm \
    -v $(pwd)/backups:/backups \
    -v $(pwd)/data/neo4j:/data \
    neo4j:5.14.0 \
    neo4j-admin database load neo4j \
    --from-path=/backups/${BACKUP_FILE}

# Redémarrer
docker-compose start neo4j

echo "Restore terminé depuis: ${BACKUP_FILE}"
```

**Health Check System :**

```bash
#!/bin/bash
# scripts/health_check.sh

echo "=== MTE System Health Check ==="

# Check Neo4j
NEO4J_STATUS=$(curl -s http://localhost:7474 | grep -o "Neo4j" | head -1)
if [ "$NEO4J_STATUS" == "Neo4j" ]; then
    echo "✅ Neo4j: Healthy"
else
    echo "❌ Neo4j: Unhealthy"
fi

# Check Greffier
GREFFIER_STATUS=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$GREFFIER_STATUS" == "healthy" ]; then
    echo "✅ Greffier: Healthy"
else
    echo "❌ Greffier: Unhealthy"
fi

# Check disk space
DISK_USAGE=$(df -h /var/lib/docker | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -lt 80 ]; then
    echo "✅ Disk: ${DISK_USAGE}% used"
else
    echo "⚠️  Disk: ${DISK_USAGE}% used (high)"
fi

# Check event count
EVENT_COUNT=$(docker exec mte-neo4j cypher-shell -u neo4j -p securepassword \
    "MATCH (e:Event) RETURN count(e) as count" --format plain | tail -1)
echo "📊 Total Events: ${EVENT_COUNT}"
```

### 10.4 Production Deployment (Phase 1+)

**Cloud Provider : AWS / GCP / Azure**

**Stack Recommandé :**
- **Compute** : AWS ECS / GCP Cloud Run / Azure Container Instances
- **Database** : Neo4j AuraDB (managed) ou Neo4j self-hosted sur EC2/GCE
- **Load Balancer** : AWS ALB / GCP Load Balancer / Azure Load Balancer
- **Storage** : AWS S3 / GCS / Azure Blob (backups)
- **Monitoring** : AWS CloudWatch / GCP Cloud Monitoring / Datadog
- **Secrets** : AWS Secrets Manager / GCP Secret Manager / Azure Key Vault

**Terraform Example (AWS) :**

```hcl
# infrastructure/main.tf
provider "aws" {
  region = "eu-west-1"
}

# ECS Cluster
resource "aws_ecs_cluster" "mte" {
  name = "mte-cluster"
}

# Task Definition (Greffier)
resource "aws_ecs_task_definition" "greffier" {
  family                   = "greffier"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = "512"
  memory                   = "1024"
  
  container_definitions = jsonencode([{
    name  = "greffier"
    image = "YOUR_ECR_REPO/greffier:latest"
    portMappings = [{
      containerPort = 8000
      protocol      = "tcp"
    }]
    environment = [
      {name = "NEO4J_URI", value = var.neo4j_uri},
      {name = "NEO4J_USER", value = var.neo4j_user}
    ]
    secrets = [{
      name      = "NEO4J_PASSWORD"
      valueFrom = aws_secretsmanager_secret.neo4j_password.arn
    }]
  }])
}

# ECS Service
resource "aws_ecs_service" "greffier" {
  name            = "greffier-service"
  cluster         = aws_ecs_cluster.mte.id
  task_definition = aws_ecs_task_definition.greffier.arn
  desired_count   = 2
  launch_type     = "FARGATE"
  
  load_balancer {
    target_group_arn = aws_lb_target_group.greffier.arn
    container_name   = "greffier"
    container_port   = 8000
  }
  
  network_configuration {
    subnets         = var.private_subnets
    security_groups = [aws_security_group.greffier.id]
  }
}

# Application Load Balancer
resource "aws_lb" "mte" {
  name               = "mte-alb"
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnets
}

# ... (target groups, listeners, security groups, etc.)
```

---

## 11. MONITORING & OBSERVABILITÉ

### 11.1 Logs Structurés

```python
# utils/logger.py
import logging
import json
from datetime import datetime
from pythonjsonlogger import jsonlogger

def setup_logging(log_level: str = "INFO", log_file: str = None):
    """Configure logging structuré JSON"""
    
    # Custom formatter
    class CustomJsonFormatter(jsonlogger.JsonFormatter):
        def add_fields(self, log_record, record, message_dict):
            super().add_fields(log_record, record, message_dict)
            log_record['timestamp'] = datetime.utcnow().isoformat()
            log_record['level'] = record.levelname
            log_record['logger'] = record.name
    
    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(CustomJsonFormatter(
        '%(timestamp)s %(level)s %(logger)s %(message)s'
    ))
    
    # Handler fichier
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(CustomJsonFormatter(
            '%(timestamp)s %(level)s %(logger)s %(message)s'
        ))
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    root_logger.addHandler(console_handler)
    if log_file:
        root_logger.addHandler(file_handler)
    
    return root_logger

# Usage
logger = logging.getLogger(__name__)
logger.info("Event logged", extra={
    "user_id": "matthias_hash",
    "event_type": "user_message",
    "node_id": "550e8400-..."
})

# Output:
# {
#   "timestamp": "2025-10-26T14:30:00.000Z",
#   "level": "INFO",
#   "logger": "greffier.main",
#   "message": "Event logged",
#   "user_id": "matthias_hash",
#   "event_type": "user_message",
#   "node_id": "550e8400-..."
# }
```

### 11.2 Métriques Application

```python
# Prometheus metrics (déjà défini Section 9.3)
from prometheus_client import make_asgi_app
from fastapi import FastAPI

app = FastAPI()

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# Métriques exposées sur http://localhost:8000/metrics
```

**Métriques Clés :**
- `api_requests_total` : Compteur requêtes API
- `api_request_duration_seconds` : Latence requêtes
- `total_events` : Nombre total événements
- `correlation_edges_total` : Nombre arêtes corrélation
- `ml_model_inference_duration_seconds` : Latence inférence ML
- `neo4j_query_duration_seconds` : Latence requêtes Neo4j

### 11.3 Tracing Distribué (Optionnel Phase 1+)

```python
from opentelemetry import trace
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.jaeger.thrift import JaegerExporter

# Setup tracer
trace.set_tracer_provider(TracerProvider())
jaeger_exporter = JaegerExporter(
    agent_host_name="localhost",
    agent_port=6831,
)
trace.get_tracer_provider().add_span_processor(
    BatchSpanProcessor(jaeger_exporter)
)

# Instrument FastAPI
FastAPIInstrumentor.instrument_app(app)

# Traces apparaissent dans Jaeger UI
```

### 11.4 Dashboards Grafana

**Dashboard Example (JSON) :**

```json
{
  "dashboard": {
    "title": "MTE System Overview",
    "panels": [
      {
        "title": "API Request Rate",
        "targets": [{
          "expr": "rate(api_requests_total[5m])"
        }],
        "type": "graph"
      },
      {
        "title": "API Latency (p95)",
        "targets": [{
          "expr": "histogram_quantile(0.95, api_request_duration_seconds_bucket)"
        }],
        "type": "graph"
      },
      {
        "title": "Total Events",
        "targets": [{
          "expr": "total_events"
        }],
        "type": "stat"
      },
      {
        "title": "Correlation Edges",
        "targets": [{
          "expr": "correlation_edges_total"
        }],
        "type": "stat"
      }
    ]
  }
}
```

### 11.5 Alerting

```yaml
# alertmanager/rules.yml
groups:
  - name: mte_alerts
    interval: 30s
    rules:
      - alert: HighAPILatency
        expr: histogram_quantile(0.95, api_request_duration_seconds_bucket) > 1.0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API latency élevée (p95 > 1s)"
          
      - alert: DatabaseDown
        expr: up{job="neo4j"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Neo4j database down"
          
      - alert: DiskSpaceHigh
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Espace disque < 10%"
```

---

## 12. TESTS

### 12.1 Tests Unitaires

```python
# greffier/tests/test_sentiment.py
import pytest
from services.sentiment import SentimentAnalyzer

@pytest.fixture
def analyzer():
    return SentimentAnalyzer()

def test_sentiment_positive(analyzer):
    text = "Je me sens incroyablement bien aujourd'hui !"
    score = analyzer.analyze(text)
    assert score > 0.5, "Sentiment devrait être positif"

def test_sentiment_negative(analyzer):
    text = "Je suis vraiment déprimé et fatigué"
    score = analyzer.analyze(text)
    assert score < -0.3, "Sentiment devrait être négatif"

def test_sentiment_neutral(analyzer):
    text = "Le chat est sur le tapis"
    score = analyzer.analyze(text)
    assert -0.2 < score < 0.2, "Sentiment devrait être neutre"

def test_empty_text(analyzer):
    score = analyzer.analyze("")
    assert score == 0.0, "Texte vide devrait retourner 0.0"
```

**Exécution :**

```bash
pytest greffier/tests/ -v --cov=greffier --cov-report=html
```

### 12.2 Tests d'Intégration

```python
# greffier/tests/test_api_integration.py
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_log_event_success():
    """Test logging événement complet"""
    event = {
        "agent": "rodin",
        "user_id": "test_user_123",
        "session_id": "test_session_456",
        "type": "user_message",
        "contenu": {
            "texte": "Message de test",
            "tags": ["test"]
        },
        "contexte": {
            "domaine": "test",
            "phase": "test_phase"
        }
    }
    
    response = client.post("/log_event", json=event)
    
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "node_id" in data
    assert "timestamp" in data

def test_log_event_validation_error():
    """Test validation erreur"""
    invalid_event = {
        "agent": "rodin",
        # Manque user_id (requis)
        "type": "user_message",
        "contenu": {"texte": "Test"},
        "contexte": {"domaine": "test"}
    }
    
    response = client.post("/log_event", json=invalid_event)
    assert response.status_code == 422

def test_health_check():
    """Test health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
```

### 12.3 Tests End-to-End

```python
# tests/test_e2e_flow.py
import pytest
from neo4j import GraphDatabase
from greffier.main import app as greffier_app
from juge.main import run_correlation_detection
from memorialiste.main import MemorialisteService

@pytest.fixture
def neo4j_driver():
    driver = GraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "testpassword")
    )
    yield driver
    # Cleanup
    with driver.session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    driver.close()

def test_complete_flow(neo4j_driver):
    """Test flux complet: Log → Détection → Enrichissement"""
    
    # 1. Log plusieurs événements
    events = [
        # Stress (sentiment négatif)
        {"type": "user_message", "texte": "Stressé par projet travail", 
         "timestamp": "2025-10-26T10:00:00Z"},
        # Craving 2h après
        {"type": "craving", "texte": "Envie forte de fumer",
         "timestamp": "2025-10-26T12:00:00Z"},
        # Répétition pattern
        {"type": "user_message", "texte": "Encore ce projet stressant",
         "timestamp": "2025-10-27T10:00:00Z"},
        {"type": "craving", "texte": "Craving again",
         "timestamp": "2025-10-27T12:00:00Z"},
        {"type": "user_message", "texte": "Projet toujours stressant",
         "timestamp": "2025-10-28T10:00:00Z"},
        {"type": "craving", "texte": "Encore une envie",
         "timestamp": "2025-10-28T12:00:00Z"}
    ]
    
    for event in events:
        # Log via API (simulation)
        pass
    
    # 2. Exécuter détection corrélations
    run_correlation_detection()
    
    # 3. Vérifier arêtes créées
    with neo4j_driver.session() as session:
        result = session.run("""
            MATCH ()-[r:CORRELATION_OBSERVEE]->()
            WHERE r.pattern_type = 'temporal_negative_to_craving'
            RETURN count(r) as correlation_count
        """)
        count = result.single()["correlation_count"]
        assert count > 0, "Au moins une corrélation devrait être détectée"
    
    # 4. Tester enrichissement contexte
    memorialiste = MemorialisteService()
    enriched_prompt = memorialiste.enrich_context(
        user_id="test_user",
        user_message="Je me sens stressé",
        session_id="test_session"
    )
    
    assert "PATTERNS OBSERVÉS" in enriched_prompt
    assert "stress" in enriched_prompt.lower() or "craving" in enriched_prompt.lower()
```

### 12.4 Tests de Charge (Phase 1+)

```python
# tests/load_test.py
from locust import HttpUser, task, between

class MTEUser(HttpUser):
    wait_time = between(1, 3)
    
    @task(3)
    def log_message(self):
        """Simule logging message (75% du traffic)"""
        self.client.post("/log_event", json={
            "agent": "rodin",
            "user_id": f"user_{self.user_id}",
            "session_id": "load_test_session",
            "type": "user_message",
            "contenu": {
                "texte": "Load test message"
            },
            "contexte": {
                "domaine": "load_test"
            }
        })
    
    @task(1)
    def health_check(self):
        """Health check (25% du traffic)"""
        self.client.get("/health")

# Exécution:
# locust -f tests/load_test.py --host=http://localhost:8000
```

---

## 13. LIMITES & CONTRAINTES

### 13.1 Limites Techniques Phase 0

**1. Causalité**
- ❌ Pas de détection causale objective
- ✅ Seulement corrélations temporelles/sémantiques
- **Impact** : Hypothèses, pas vérités

**2. Prédiction**
- ❌ Pas de système prédictif Phase 0
- **Raison** : Nécessite données matures (90+ jours)
- **Workaround** : Observation patterns a posteriori

**3. Évaluation Efficacité**
- ❌ Pas de mesure automatique ROI interventions
- **Raison** : Subjectivité, complexité mesure
- **Workaround** : Feedback qualitatif utilisateur

**4. Multi-Utilisateurs**
- ❌ Optimisé pour single user
- **Raison** : Volumétrie faible, complexité inutile
- **Migration Phase 1** : Ajout sharding, isolation données

**5. Temps Réel**
- ⚠️ Détection corrélations = batch nocturne
- **Latence** : 24h max pour nouveaux patterns
- **Workaround Phase 1** : Détection incrémentale temps réel

### 13.2 Limites Conceptuelles

**1. Problème Confounders (Variables Cachées)**

```
Exemple:
- Événement A : "Stress travail" (observé)
- Événement B : "Craving" (observé)
- Variable X : "Manque sommeil" (non observé)

Réalité : X → A et X → B
Système détecte : A → B (faux)
```

**Mitigation :**
- Disclaimer utilisateur : patterns = observations, pas causalité
- Validation humaine requise
- Phase 1+ : Capturer plus de variables (sommeil, météo, etc.)

**2. Biais Sélection (Selection Bias)**

L'utilisateur log événements notables (douleur, craving) mais pas événements normaux (moments bien-être). Le système sur-détecte patterns négatifs.

**Mitigation :**
- Encourager logging exhaustif (bon et mauvais)
- Reminders automatiques (Phase 1+)
- Balance positif/négatif dans UI

**3. Biais Confirmation**

L'utilisateur peut confirmer patterns suggérés même si non-pertinents (power of suggestion).

**Mitigation :**
- Questions neutres ("Est-ce lié ?", pas "C'est lié, non ?")
- Permettre invalidation explicite
- Tracker taux confirmation (si trop élevé, suspicion biais)

### 13.3 Contraintes Opérationnelles

**Phase 0 :**
- Single user (Matthias)
- Déploiement local uniquement
- Pas d'authentification robuste
- Pas de backup automatique
- Support communautaire (pas commercial)

**Évolution Phase 1+ :**
- Multi-users avec isolation
- Cloud deployment
- Auth/authz robuste
- Backups automatiques quotidiens
- Support professionnel potentiel

### 13.4 Conformité & Légal

**Non-applicable Phase 0 (usage personnel) :**
- Pas de certification médicale requise
- Pas d'audit RGPD formel
- Pas d'assurance responsabilité

**Requis Phase 1+ (production publique) :**
- **Si prétentions thérapeutiques** :
  - Certification dispositif médical (FDA/CE marking)
  - Études cliniques
  - Responsabilité juridique
- **RGPD** :
  - DPO (Data Protection Officer)
  - DPIA (Data Protection Impact Assessment)
  - Conformité Article 22 (décisions automatisées)
- **Assurances** :
  - Erreurs & omissions
  - Cyber-sécurité
  - Responsabilité produit

---

## 14. ROADMAP TECHNIQUE

### 14.1 Phase 0 (J0-J90) - POC ✅

**Objectif :** Valider faisabilité technique

**Livrables :**
- ✅ Infrastructure fonctionnelle
- ✅ Greffier (ingestion événements)
- ✅ Juge V.E. (détection corrélations basiques)
- ✅ Mémorialiste V.E. (enrichissement contexte)
- ✅ 90 jours données réelles (Matthias sevrage)
- ✅ Validation subjective ("IA se souvient ?")

**Succès :** Graphe peuplé, patterns détectés, utilité perçue

---

### 14.2 Phase 1 (J90-J180) - Extension Multi-Cas

**Objectif :** Tester généralisation même domaine

**Scope :**
- 10-20 utilisateurs
- Domaine : Santé mentale (sevrage, anxiété, dépression)
- GraphRAG partagé anonymisé (optional)
- Déploiement cloud (AWS/GCP)

**Nouveautés Techniques :**
- Authentication (JWT)
- Multi-tenancy (isolation données)
- API rate limiting
- Backups automatiques
- Monitoring production (Datadog/New Relic)
- Détection corrélations incrémentale (pas full batch)

**Livrables :**
- Validation généralisation cross-utilisateurs
- Patterns universels identifiés
- Optimisations performance
- Interface web basique (dashboard utilisateur)

---

### 14.3 Phase 2 (J180-J365) - Multi-Domaines

**Objectif :** Élargir à domaines variés

**Scope :**
- 50-100 utilisateurs
- 3-5 domaines : Santé, Éducation, Business
- Multi-agents (Rodin, Lumi, Nova)
- Features avancées (prédictions, recommandations)

**Nouveautés Techniques :**
- **Prédicateur** (Prediction Engine)
  - Modèles séries temporelles (Prophet, LSTM)
  - Prédictions probabilistes (Monte Carlo)
  - Fenêtres temporelles prédiction
- **Évaluateur** (Feedback Loop)
  - A/B testing interventions
  - Reinforcement learning
  - Mise à jour force arêtes causales
- **GraphRAG Avancé**
  - Agrégation patterns cross-domaines
  - Transfer learning corrélations
  - Détection méta-patterns
- **Scalabilité**
  - Neo4j Causal Cluster (multi-nodes)
  - Horizontal scaling API (load balancer)
  - Redis cache distribué

**Livrables :**
- Validation méta-patterns transversaux
- Système robuste et scalable
- APIs externes (intégrations tierces)
- Dashboard analytics avancé

---

### 14.4 Phase 3 (J365+) - Productisation

**Objectif :** Produit commercial ou open-source

**Scope :**
- 1000+ utilisateurs
- Tous domaines
- Interface utilisateur complète
- Mobile apps (iOS/Android)
- Intégrations (Slack, Notion, Zapier, etc.)

**Nouveautés Techniques :**
- **Mobile Apps** (React Native / Flutter)
- **Voice Input** (Whisper API pour logging vocal)
- **Edge Deployment** (on-device ML pour privacy)
- **Federated Learning** (apprentissage distribué sans centralisation données)
- **Explainability** (visualisation graphes causaux interactifs)
- **Marketplace Patterns** (utilisateurs partagent patterns anonymisés)

**Modèles Business :**
- SaaS (B2C : $20-50/mois, B2B : $500+/mois)
- API-First (platform as a service)
- Open-Source + Services (hosting, support, enterprise features)
- Licensing B2B (healthcare systems, edtech platforms)

**Certifications :**
- ISO 27001 (sécurité)
- SOC 2 Type II
- HIPAA (si santé US)
- HDS (hébergement données santé France)
- FDA/CE marking (si dispositif médical)

**Livrables :**
- Plateforme production scalable
- Documentation complète (dev + user)
- Communauté (forums, Discord, etc.)
- Partenariats stratégiques
- Impact mesurable (études utilisateurs, témoignages)

---

## 15. CONCLUSION

### 15.1 Récapitulatif Architecture

Le Système de Mémoire Temporelle Empathique (MTE) est une architecture logicielle en 3 couches :

**Couche Ingestion (Greffier) :**
- Capture exhaustive événements utilisateur-agent-environnement
- Enrichissement automatique (sentiment, embeddings)
- Persistance Neo4j avec validation robuste

**Couche Analyse (Juge V.E.) :**
- Détection corrélations temporelles, sémantiques, séquentielles
- Batch nocturne (Phase 0), incrémental temps réel (Phase 1+)
- Stockage patterns dans graphe (arêtes CORRELATION_OBSERVEE)

**Couche Contextualisation (Mémorialiste V.E.) :**
- Récupération contexte pertinent (temporel, sémantique, patterns)
- Construction prompt empathique enrichi
- Intégration LLM (Claude) pour réponses mémoriales

**Principe Fondamental :**
Le système ne découvre PAS de causalité objective. Il détecte des corrélations et les présente comme hypothèses à valider par l'utilisateur, créant ainsi une continuité relationnelle empathique.

### 15.2 Points Forts

✅ **Architecture Simple & Robuste (Phase 0)**
- Composants clairement séparés
- Stack mature (FastAPI, Neo4j, HuggingFace)
- Validation technique progressive

✅ **Scalabilité Intégrée (Design)**
- Architecture extensible (Phase 1-3)
- Neo4j permet scaling horizontal
- Modularité permet évolution incrémentale

✅ **Privacy by Design**
- Anonymisation user_id
- Données sensibles isolées
- Conformité RGPD préparée

✅ **Observabilité**
- Logs structurés JSON
- Métriques Prometheus
- Tracing distribué (optionnel)

### 15.3 Points d'Attention

⚠️ **Limites Causales Intrinsèques**
- Confounders inévitables
- Validation humaine toujours nécessaire
- Disclaimers critiques pour utilisateurs

⚠️ **Complexité ML**
- Modèles sentiment/embedding nécessitent ressources
- Temps inférence peut impacter latence
- Besoin monitoring drift modèles (Phase 1+)

⚠️ **Dette Technique Potentielle**
- Simplifications Phase 0 devront être revues
- Migrations database (schéma évolution)
- Refactoring nécessaire scaling

### 15.4 Prochaines Étapes Immédiates

**Pour Matthias :**
1. **Décision GO/NO-GO** : Implémenter pendant sevrage ?
2. **Si GO** : Choix méthode logging (CLI, bot, autre)
3. **Setup** : Installation client logger, premiers tests

**Pour Rodin (moi) :**
1. **Ce weekend** : Setup infrastructure (Neo4j, API Greffier)
2. **Tests** : Validation pipeline complet
3. **Documentation** : Guide utilisateur client logger

**Pour Lumi :**
1. **Validation** : Review finale architecture
2. **Paramètres** : Finaliser seuils (threshold, fenêtres)
3. **Script** : Juge V.E. prêt à déployer

---

## ANNEXES

### A. Glossaire

- **Arête** : Relation orientée entre deux nœuds dans graphe
- **Bolt Protocol** : Protocol communication Neo4j
- **Causal Inference** : Inférence causale (A cause B)
- **Correlation** : Co-occurrence statistique (A et B ensemble)
- **Embedding** : Représentation vectorielle texte
- **Event Sourcing** : Pattern architecture (historique événements immuables)
- **GraphRAG** : Retrieval-Augmented Generation avec graphe
- **LLM** : Large Language Model
- **Nœud** : Entité dans graphe (Event)
- **Sentiment Analysis** : Analyse automatique sentiment texte

### B. Ressources

**Documentation :**
- Neo4j : https://neo4j.com/docs/
- FastAPI : https://fastapi.tiangolo.com/
- HuggingFace Transformers : https://huggingface.co/docs/transformers/
- SentenceTransformers : https://www.sbert.net/

**Papers Académiques :**
- "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (Lewis et al., 2020)
- "Causal Inference in Statistics" (Pearl, 2016)
- "Granger Causality" (Granger, 1969)

**Outils :**
- Neo4j Desktop : https://neo4j.com/download/
- Postman (API testing) : https://www.postman.com/
- Grafana : https://grafana.com/

### C. Contact & Support

**Phase 0 :**
- Support communautaire (Matthias + équipe Rodin/Lumi/Nova)
- Issues tracking : GitHub (si open-sourced)
- Documentation : Ce document + README.md

**Phase 1+ :**
- Email support : support@mte.example.com
- Discord communauté : discord.gg/mte
- Documentation complète : docs.mte.example.com
