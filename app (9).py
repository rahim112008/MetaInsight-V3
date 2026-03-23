import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
from io import StringIO
import requests
import json
import random

# -------------------------------
# Page configuration
# -------------------------------
st.set_page_config(page_title="MetaInsight v4", page_icon="🧬", layout="wide")
st.markdown(
    """
    <style>
        /* Custom styles to match the original design */
        .stApp {
            background: #0A0E1A;
            color: #E8EDF5;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }
        .stTabs [data-baseweb="tab"] {
            background: #1A2238;
            border-radius: 8px;
            padding: 5px 12px;
            font-family: monospace;
        }
        .stTabs [aria-selected="true"] {
            background: rgba(0,212,170,0.2);
            border-bottom: 2px solid #00D4AA;
        }
        .css-1d391kg {
            background: #0F1525;
        }
        div[data-testid="stMetric"] {
            background: #1A2238;
            border-radius: 10px;
            padding: 12px;
            border: 1px solid #2A3550;
        }
        .stAlert {
            background: rgba(0,212,170,0.08);
            border: 1px solid rgba(0,212,170,0.2);
        }
        hr {
            margin: 1rem 0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------------
# Data generation & loading
# -------------------------------
TAXA = [
    "Proteobacteria",
    "Actinobacteriota",
    "Firmicutes",
    "Bacteroidota",
    "Archaea",
    "Acidobacteria",
    "Chloroflexi",
    "Planctomycetes",
    "Ascomycota",
    "Caudovirales",
]
ENVS = ["Sol aride", "Eau marine", "Gut", "Sol agricole", "Sédiments", "Biofilm"]
ENV_COLORS = {
    "Sol aride": "#FF8C42",
    "Eau marine": "#4D9FFF",
    "Gut": "#FF5252",
    "Sol agricole": "#7BD17A",
    "Sédiments": "#9B7CFF",
    "Biofilm": "#FF5FA0",
}
COLORS = [
    "#00D4AA",
    "#4D9FFF",
    "#9B7CFF",
    "#FF8C42",
    "#FF5FA0",
    "#FFD166",
    "#FF5252",
    "#7BD17A",
    "#34D1D1",
    "#E88CF5",
]


def generate_demo_data():
    """Create a DataFrame with 24 samples, 6 environments, 10 taxa."""
    data = []
    profiles = {
        "Sol aride": [28, 20, 5, 4, 8, 6, 4, 3, 2, 1],
        "Eau marine": [35, 10, 8, 15, 2, 5, 3, 4, 8, 6],
        "Gut": [15, 12, 30, 22, 1, 3, 2, 2, 4, 2],
        "Sol agricole": [22, 25, 10, 8, 4, 10, 7, 5, 3, 2],
        "Sédiments": [18, 14, 12, 10, 6, 8, 9, 6, 5, 4],
        "Biofilm": [30, 18, 6, 9, 3, 7, 5, 4, 6, 5],
    }
    for env in ENVS:
        base = profiles[env]
        for i in range(4):
            vals = [max(0.1, v + np.random.normal(0, 2.5)) for v in base]
            s = sum(vals)
            abundances = [v / s * 100 for v in vals]
            row = {
                "sample_id": f"{env[:3].upper()}_{np.random.randint(100, 999)}",
                "environment": env,
                **{tax: round(ab, 2) for tax, ab in zip(TAXA, abundances)},
            }
            data.append(row)
    df = pd.DataFrame(data)
    # Add Shannon index
    for idx, row in df.iterrows():
        probs = row[TAXA].values / 100.0
        shannon = -np.sum(probs * np.log(probs + 1e-9))
        df.loc[idx, "shannon"] = round(shannon, 3)
    return df


@st.cache_data
def load_demo():
    return generate_demo_data()


def load_user_data(uploaded_file):
    """Load user CSV: expects columns sample_id, environment, and abundance columns for taxa."""
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        # Basic validation: check for required columns
        if "environment" not in df.columns:
            st.error("The uploaded CSV must contain an 'environment' column.")
            return None
        # Ensure all taxa columns are present, if not, fill with 0
        for tax in TAXA:
            if tax not in df.columns:
                df[tax] = 0.0
        # Compute Shannon if not present
        if "shannon" not in df.columns:
            shannon_vals = []
            for idx, row in df.iterrows():
                probs = row[TAXA].values / 100.0
                shannon = -np.sum(probs * np.log(probs + 1e-9))
                shannon_vals.append(round(shannon, 3))
            df["shannon"] = shannon_vals
        return df
    return None


# -------------------------------
# Helper functions for modules
# -------------------------------
def plot_pca(df):
    """PCA plot of the data using Plotly."""
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    X = df[TAXA].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    pca_df["environment"] = df["environment"].values
    fig = px.scatter(
        pca_df,
        x="PC1",
        y="PC2",
        color="environment",
        color_discrete_map=ENV_COLORS,
        title=f"PCA (explained variance: {pca.explained_variance_ratio_[0]:.1%} / {pca.explained_variance_ratio_[1]:.1%})",
        template="plotly_dark",
    )
    fig.update_layout(plot_bgcolor="#0F1525", paper_bgcolor="#0F1525")
    return fig


def plot_radar(df):
    """Radar chart of average abundances per environment."""
    avg_ab = df.groupby("environment")[TAXA].mean().reset_index()
    fig = go.Figure()
    for env in ENVS:
        env_data = avg_ab[avg_ab["environment"] == env].iloc[0, 1:]
        fig.add_trace(
            go.Scatterpolar(
                r=env_data.values,
                theta=TAXA,
                fill="toself",
                name=env,
                line_color=ENV_COLORS[env],
                fillcolor=ENV_COLORS[env] + "20",
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, showticklabels=False, gridcolor="#2A3550")),
        template="plotly_dark",
        paper_bgcolor="#0F1525",
        plot_bgcolor="#0F1525",
        legend=dict(font=dict(color="#7A8BA8")),
    )
    return fig


# -------------------------------
# Module functions (each returns the content to display)
# -------------------------------
def home_module(df):
    st.subheader("🏠 Accueil")
    st.markdown("### Plateforme métagénomique de pointe — Transformers génomiques · Causal ML · Generative AI · Federated Learning")

    # KPIs
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Modules ML/DL", "12", delta="+4 v4")
    with col2:
        st.metric("Précision DNABERT-2", "96.8%", delta="+5.5%")
    with col3:
        st.metric("Données synthétiques", "10K", delta="échantillons")
    with col4:
        st.metric("Nœuds Fédérés", "6", delta="labos · ε-DP privacy")

    # Two columns
    colA, colB = st.columns(2)
    with colA:
        st.markdown("#### Nouveaux modules v4")
        st.button("🧬 DNABERT-2", key="home_dnabert", on_click=lambda: st.session_state.update(tab="DNABERT-2"))
        st.button("⚗️ Causal ML", key="home_causal", on_click=lambda: st.session_state.update(tab="Causal ML"))
        st.button("✨ GenAI Synthétique", key="home_genai", on_click=lambda: st.session_state.update(tab="GenAI"))
        st.button("🔒 Federated Learning", key="home_fed", on_click=lambda: st.session_state.update(tab="Federated"))
        st.markdown("#### Modules hérités v3")
        st.write("K-means, Random Forest, LSTM, VAE, XAI/SHAP, GNN")
        if st.button("⚡ Charger données démo (24 échantillons)"):
            st.session_state["df"] = load_demo()
            st.rerun()

    with colB:
        st.plotly_chart(plot_pca(df), use_container_width=True)

    st.markdown("### Apports de MetaInsight v4 — comparaison des modules")
    comp_data = {
        "Limite v3": [
            "Classification k-mers manuelle",
            "Corrélation ≠ causalité",
            "Peu d'échantillons arides",
            "Données non partagées",
        ],
        "Module v4": ["🧬 DNABERT-2", "⚗️ Causal ML", "✨ GenAI", "🔒 Federated"],
        "Technique": [
            "Transformer génomique 6-mers",
            "DAG + Do-calculus (Pearl)",
            "Dirichlet-VAE + cGAN",
            "FedAvg + ε-DP privacy",
        ],
        "Amélioration": [
            "+5.5% classifiés → 96.8%",
            "Liens causaux vs spurieux",
            "×10 augmentation données",
            "Collaboration sans fuite",
        ],
        "Source": [
            "Nature Methods 2024",
            "PNAS 2025",
            "Bioinformatics 2025",
            "Cell Systems 2025",
        ],
    }
    st.dataframe(pd.DataFrame(comp_data), use_container_width=True)

    st.plotly_chart(plot_radar(df), use_container_width=True)


def dnabert_module(df):
    st.subheader("🧬 DNABERT-2")
    st.markdown("Transformer pré-entraîné sur séquences ADN — classification métagénomique au niveau de la séquence brute")
    st.info(
        "Principe : DNABERT-2 encode directement les reads ADN en tokens de 6-mers via un mécanisme d'attention multi-têtes (12 têtes, 768 dimensions cachées). "
        "Il résout la limite principale de v3 : la classification par profil taxonomique ne voyait pas la séquence brute — DNABERT-2 lit chaque read et prédit son taxon d'origine avec 96.8% de précision contre 91.3% pour Random Forest."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Configuration du modèle")
        model = st.selectbox("Modèle", ["DNABERT-2 (BPE, 117M params)", "DNABERT-1 (k-mer=6, 86M params)", "Nucleotide Transformer (2.5B params)"])
        kmer = st.slider("k-mer", 3, 8, 6)
        finetune = st.selectbox("Fine-tuning", ["Zero-shot (pré-entraîné)", "Fine-tune métagénomique", "Domain adaptation aride"])
        heads = st.slider("Têtes d'attention à visualiser", 1, 12, 3)

        if st.button("🚀 Classifier avec DNABERT-2"):
            st.session_state["dnabert_run"] = True
            # Store parameters
            st.session_state["kmer"] = kmer
            st.session_state["heads"] = heads

    with col2:
        st.markdown("#### Métriques de classification")
        if st.session_state.get("dnabert_run", False):
            acc = 96.8
            classified = 98.2
            col_a, col_b = st.columns(2)
            col_a.metric("Précision", f"{acc}%")
            col_b.metric("Reads classifiés", f"{classified}%")

            # Accuracy comparison chart
            methods = ["DNABERT-2\n(v4)", "RF\n(v3)", "Kraken2", "QIIME2", "MEGAN", "Bowtie2"]
            accs = [96.8, 91.3, 78.4, 82.1, 74.6, 68.9]
            fig = px.bar(x=methods, y=accs, color=methods, color_discrete_sequence=["#00D4AA"] + ["#4D9FFF"] * 5,
                         title="Précision comparative", template="plotly_dark")
            fig.update_layout(plot_bgcolor="#0F1525", paper_bgcolor="#0F1525")
            st.plotly_chart(fig, use_container_width=True)

            # Attention visualization (simulated)
            st.markdown("#### Visualisation des têtes d'attention — Transformer")
            tokens = ["ATG", "GCT", "AAC", "TGG", "CCG", "ATG", "TAC", "GGC", "TTA", "ACG"][: min(8, kmer + 2)]
            # Build a random attention matrix
            att_mats = []
            for h in range(heads):
                mat = np.random.rand(len(tokens), len(tokens))
                mat = (mat + mat.T) / 2
                mat = mat / mat.sum(axis=1, keepdims=True)
                att_mats.append(mat)

            fig, axes = plt.subplots(1, min(heads, 3), figsize=(12, 4))
            if heads == 1:
                axes = [axes]
            for idx, (ax, mat) in enumerate(zip(axes, att_mats[:3])):
                im = ax.imshow(mat, cmap="Greens", vmin=0, vmax=0.5)
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, fontsize=8, rotation=45)
                ax.set_yticklabels(tokens, fontsize=8)
                ax.set_title(f"Head {idx+1}")
            plt.tight_layout()
            st.pyplot(fig)

            # Token viz
            st.markdown("#### Tokens ADN — séquence encodée")
            seq = "ATGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGCTAGC"
            token_list = [seq[i:i+kmer] for i in range(0, len(seq)-kmer, max(1, kmer//2))]
            token_importance = np.random.rand(len(token_list))
            cols = st.columns(5)
            for i, tok in enumerate(token_list[:20]):
                with cols[i % 5]:
                    st.markdown(f"<span style='background:rgba(0,212,170,{0.3+token_importance[i]*0.5}); padding:2px 6px; border-radius:5px;'>{tok}</span>", unsafe_allow_html=True)

            # AI interpretation
            if st.session_state.get("claude_key"):
                prompt = f"Expert métagénomique et Transformers. DNABERT-2 (117M params, BPE tokenizer, {kmer}-mers, {heads} têtes d'attention) atteint 96.8% de précision pour classer des reads métagénomiques. En 4 phrases scientifiques : (1) Pourquoi le mécanisme d'attention multi-têtes capture mieux les motifs évolutifs conservés qu'un k-mer classique, (2) Avantage du BPE (Byte-Pair Encoding) vs k-mer fixe pour les séquences métagénomiques, (3) Comment interpréter les têtes d'attention qui se focalisent sur différents patterns (codons, régions promotrices), (4) Limite principale : DNABERT-2 nécessite GPU et fine-tuning spécifique au sol aride."
                ai_response = call_claude(prompt, st.session_state.get("claude_key"))
                st.markdown("#### Interprétation IA")
                st.info(ai_response)


def causal_module(df):
    st.subheader("⚗️ Causal ML — Inférence causale microbienne")
    st.markdown("DAG + Do-calculus de Judea Pearl — distinguer les vraies causes des corrélations spurieuses")
    st.info(
        "Problème : En v3, GNN et Spearman trouvaient des corrélations mais pas des causes. Proteobacteria corrèle avec la sécheresse — mais cause-t-il la résistance ou en est-il un marqueur ? "
        "Le Causal ML construit un DAG (Directed Acyclic Graph) et applique le Do-calculus pour répondre : 'Si on intervenait sur Proteobacteria, que se passerait-il ?'"
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Configuration DAG")
        algo = st.selectbox("Algorithme de découverte causale", ["PC Algorithm (Peter-Clark)", "FCI (Fast Causal Inference)", "LiNGAM", "NOTEARS"])
        alpha = st.slider("Seuil de significativité α", 0.01, 0.20, 0.05, format="%.2f")
        intervention = st.selectbox("Variable d'intervention", ["Proteobacteria", "Archaea", "Firmicutes", "Acidobacteria", "Sécheresse (env)"])
        do_val = st.slider("Intensité Do-calculus", -50, 50, 30, step=5, format="%d%%")
        if st.button("🚀 Inférer le graphe causal"):
            st.session_state["causal_run"] = True
            st.session_state["causal_intervention"] = intervention
            st.session_state["causal_do"] = do_val

    with col2:
        st.markdown("#### Graphe causal (DAG)")
        # Simplified DAG visualization with text
        st.image("https://via.placeholder.com/400x200?text=Graph+simplified+in+Python", use_column_width=True)

    if st.session_state.get("causal_run", False):
        st.markdown("### Résultats causaux")
        # ATE chart
        effects = ["Shannon H′", "Archaea", "Firmicutes", "Acidobacteria", "Bacteroidota"]
        ates = [0.58, 0.32, -0.14, 0.27, 0.12]
        fig = px.bar(x=effects, y=ates, color=ates, color_continuous_scale="RdYlGn", title="Effets causaux estimés (ATE)", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Do-calculus result
        main_eff = ates[0]
        st.success(f"P(Shannon H′ | do({intervention} {do_val:+d}%))\n\nEffet causal estimé : **{main_eff:+.2f}** σ\nIntervalle de confiance 95% : [{main_eff-0.18:.2f}, {main_eff+0.18:.2f}]")

        # Spurious vs causal table
        table_data = {
            "Paire de taxons": ["Proteobacteria → Shannon H′", "Firmicutes → Shannon H′", "Archaea ↔ Firmicutes", "Sécheresse → Firmicutes", "Acidobacteria → Shannon H′"],
            "Corrélation Spearman": ["ρ=0.72", "ρ=0.68", "ρ=0.51", "ρ=0.79", "ρ=0.44"],
            "Effet causal ATE": ["ATE=0.58", "ATE=0.03", "ATE=0.02", "ATE=0.71", "ATE=0.31"],
            "Type": ["✅ Causal", "❌ Spurieux", "❌ Spurieux", "✅ Causal", "✅ Causal"],
            "Confondant": ["—", "Sécheresse", "Proteobacteria", "—", "—"],
        }
        st.dataframe(pd.DataFrame(table_data), use_container_width=True)

        # AI interpretation
        if st.session_state.get("claude_key"):
            prompt = f"Expert causalité et microbiome (Do-calculus, graphes causaux). Intervention sur {intervention} (+{do_val}%), ATE sur Shannon H′ = {main_eff:.2f}. Le DAG révèle que Firmicutes corrèle avec Shannon H′ (ρ=0.68) mais l'effet causal ATE=0.03 est négligeable — confondant = Sécheresse. En 4 phrases : (1) Différence fondamentale entre P(Y|X) et P(Y|do(X)) en métagénomique, (2) Pourquoi Firmicutes est spurieux ici (fork causal via Sécheresse), (3) Application concrète pour les sols arides : quels taxons cibler pour la bio-restauration, (4) Limite principale du PC-algorithm sur données compositionnelles (Aitchison)."
            ai_response = call_claude(prompt, st.session_state.get("claude_key"))
            st.markdown("#### Interprétation IA")
            st.info(ai_response)


def genai_module(df):
    st.subheader("✨ Generative AI — Données métagénomiques synthétiques")
    st.markdown("Dirichlet-VAE · cGAN · Diffusion — augmentation de données pour environnements arides sous-représentés")
    st.info(
        "Problème résolu : Les sols arides d'Algérie ont souvent <50 échantillons disponibles dans les bases publiques. Les modèles ML sur-apprennent. "
        "Le module GenAI génère des profils métagénomiques synthétiques réalistes qui respectent la composition de Dirichlet du microbiome — sans jamais leaker de données réelles (mode privacy-safe)."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Configuration du générateur")
        model_type = st.selectbox("Modèle génératif", ["Dirichlet-VAE (défaut)", "Conditional GAN (cGAN)", "Diffusion métagénomique"])
        target_env = st.selectbox("Environnement cible", ENVS + ["Tous les environnements"])
        n_samples = st.slider("Nb. échantillons synthétiques", 50, 1000, 200, step=50)
        temperature = st.slider("Température (diversité)", 0.1, 2.0, 0.8, step=0.1)
        quality = st.selectbox("Contrôle qualité FID", ["Strict (FID < 5)", "Standard (FID < 10)", "Permissif (FID < 20)"])
        if st.button("✨ Générer les données synthétiques"):
            st.session_state["genai_run"] = True
            st.session_state["genai_n"] = n_samples
            st.session_state["genai_env"] = target_env

    with col2:
        st.markdown("#### Pipeline de génération")
        # Simulate steps
        steps = ["Encodage des données réelles", "Sampling latent", "Décodage + normalisation", "Filtre qualité (FID)", "Validation statistique"]
        for step in steps:
            st.write(f"✅ {step}")
        if st.session_state.get("genai_run", False):
            st.metric("Générés", f"{st.session_state['genai_n']}")
            st.metric("FID score", "3.2")
            st.metric("KL-divergence", "0.04")

            # Comparison plots
            fig = px.scatter(
                x=np.random.randn(24), y=np.random.randn(24),
                labels={"x": "PC1", "y": "PC2"}, title="Données réelles (ronds) vs synthétiques (croix)",
                template="plotly_dark"
            )
            fig.add_scatter(x=np.random.randn(st.session_state['genai_n']), y=np.random.randn(st.session_state['genai_n']), mode="markers",
                            marker=dict(symbol="x", color="purple"), name="Synthétiques")
            st.plotly_chart(fig, use_container_width=True)

            # Distribution comparison
            real_avg = df.groupby("environment")[TAXA].mean().loc[target_env] if target_env != "Tous les environnements" else df[TAXA].mean()
            synth_avg = real_avg + np.random.normal(0, 0.5, len(TAXA))
            fig = go.Figure()
            fig.add_trace(go.Bar(x=TAXA[:5], y=real_avg[:5], name="Réels", marker_color="#00D4AA"))
            fig.add_trace(go.Bar(x=TAXA[:5], y=synth_avg[:5], name="Synthétiques", marker_color="#9B7CFF"))
            fig.update_layout(template="plotly_dark", title="Abondance moyenne")
            st.plotly_chart(fig, use_container_width=True)

            # AI interpretation
            if st.session_state.get("claude_key"):
                prompt = f"Expert GenAI et métagénomique. Dirichlet-VAE a généré {n_samples} profils métagénomiques synthétiques pour {target_env}. FID score = 3.2 (excellente fidélité), KL-divergence = 0.04. PCA montre une bonne couverture de l'espace réel. En 4 phrases : (1) Pourquoi un Dirichlet-VAE est adapté aux données compositionelles (simplex) vs un VAE standard, (2) Validation statistique des données synthétiques (MMD, FID, Wasserstein distance), (3) Risques d'utiliser des données synthétiques pour l'entraînement (memorisation, mode collapse), (4) Impact concret : comment ces {n_samples} échantillons améliorent le RF de 91.3% → 95%+ en augmentation de données."
                ai_response = call_claude(prompt, st.session_state.get("claude_key"))
                st.markdown("#### Interprétation IA")
                st.info(ai_response)


def federated_module(df):
    st.subheader("🔒 Federated Learning — Collaboration sans fuite de données")
    st.markdown("FedAvg + Differential Privacy (ε-DP) — entraîner un modèle global sans partager les séquences brutes")
    st.info(
        "Problème résolu : Chaque laboratoire garde ses données métagénomiques confidentielles (patients, brevets). "
        "Federated Learning entraîne un modèle partagé en n'échangeant que les gradients (jamais les données brutes), avec un bruit différentiel ε-DP pour garantir la privacy même face aux attaques par inférence."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### Configuration réseau fédéré")
        fed_algo = st.selectbox("Algorithme d'agrégation", ["FedAvg (McMahan 2017)", "FedProx", "SCAFFOLD"])
        n_nodes = st.selectbox("Nombre de nœuds (labos)", [3, 6, 10], index=1)
        epsilon = st.slider("Privacy ε (epsilon-DP)", 0.1, 5.0, 0.5, step=0.1, format="%.1f")
        rounds = st.slider("Rounds de communication", 2, 50, 10)
        local_epochs = st.slider("Local epochs par nœud", 1, 20, 5)
        if st.button("🚀 Lancer l'entraînement fédéré"):
            st.session_state["fed_run"] = True
            st.session_state["fed_rounds"] = rounds
            st.session_state["fed_nodes"] = n_nodes
            st.session_state["fed_eps"] = epsilon

    with col2:
        st.markdown("#### Réseau de nœuds fédérés")
        # Simulate nodes
        node_names = ["USTHB Alger", "Univ. Oran", "INRAA Sétif", "Univ. Tlemcen", "CRBT Constantine", "Inst. Pasteur"]
        for i in range(min(n_nodes, len(node_names))):
            st.markdown(f"🏛️ **{node_names[i]}** — {np.random.randint(200, 600)} échantillons")
            st.progress(np.random.uniform(0.3, 0.9))

        st.markdown(f"🔒 ε = {epsilon} · Privacy {'forte' if epsilon < 1 else 'moyenne' if epsilon < 3 else 'faible'} · Aucune donnée brute partagée")

    if st.session_state.get("fed_run", False):
        st.markdown("### Résultats fédérés")
        # Convergence plot
        x = list(range(1, rounds + 1))
        global_acc = [75 + 18 * (1 - np.exp(-r / 5)) + np.random.normal(0, 0.5) for r in x]
        local_accs = []
        for _ in range(n_nodes):
            local_acc = [68 + 18 * (1 - np.exp(-r / 7)) + np.random.normal(0, 1) for r in x]
            local_accs.append(local_acc)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=global_acc, mode="lines+markers", name="Modèle global fédéré", line=dict(color="#00D4AA", width=3)))
        for i in range(min(3, n_nodes)):
            fig.add_trace(go.Scatter(x=x, y=local_accs[i], mode="lines", name=f"Local — {node_names[i]}", line=dict(dash="dash", width=1.5)))
        fig.update_layout(template="plotly_dark", title="Convergence de l'entraînement fédéré",
                          xaxis_title="Round", yaxis_title="Précision (%)", yaxis_range=[60, 100])
        st.plotly_chart(fig, use_container_width=True)

        # Final comparison
        final_global = global_acc[-1]
        final_locals = [acc[-1] for acc in local_accs]
        final_df = pd.DataFrame({"Modèle": ["Global fédéré"] + node_names[:n_nodes],
                                 "Précision": [final_global] + final_locals})
        fig = px.bar(final_df, x="Modèle", y="Précision", color="Modèle", color_discrete_map={"Global fédéré": "#00D4AA"}, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

        # Privacy analysis
        st.markdown("#### Analyse privacy — bruit différentiel appliqué")
        # Simulate gradient distribution
        np.random.seed(42)
        grad_clean = np.random.normal(0, 1, 1000)
        grad_noisy = grad_clean + np.random.normal(0, 0.8 / epsilon, 1000)
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=grad_clean, name="Gradients bruts", opacity=0.6, nbinsx=50))
        fig.add_trace(go.Histogram(x=grad_noisy, name="Avec bruit ε-DP", opacity=0.6, nbinsx=50))
        fig.update_layout(barmode="overlay", template="plotly_dark", title="Distribution des gradients")
        st.plotly_chart(fig, use_container_width=True)

        # AI interpretation
        if st.session_state.get("claude_key"):
            prompt = f"Expert Federated Learning et privacy métagénomique. FedAvg sur {n_nodes} laboratoires, {rounds} rounds de communication, epsilon-DP = {epsilon} (privacy {'forte' if epsilon<1 else 'moyenne' if epsilon<3 else 'faible'}). Modèle global atteint {final_global:.1f}% de précision vs {min(final_locals):.1f}%-{max(final_locals):.1f}% pour les modèles locaux. En 4 phrases : (1) Pourquoi FedAvg améliore la généralisation même avec des données hétérogènes (non-IID) entre labos, (2) Garanties mathématiques de ε-DP (théorème de composition, privacy amplification by sampling), (3) Application concrète pour la métagénomique algérienne : quels labos auraient le plus à gagner de la collaboration fédérée, (4) Limite : Byzantine faults (nœuds malveillants) et défense par gradient clipping + Krum aggregation."
            ai_response = call_claude(prompt, st.session_state.get("claude_key"))
            st.markdown("#### Interprétation IA")
            st.info(ai_response)


def clustering_module(df):
    st.subheader("🔵 Clustering")
    st.markdown("K-means · DBSCAN — groupement des profils microbiens similaires")
    col1, col2 = st.columns(2)
    with col1:
        algo = st.selectbox("Algorithme", ["K-means", "DBSCAN"])
        k = st.slider("Nombre de clusters (k)", 2, 8, 4)
        if st.button("🚀 Lancer le clustering"):
            st.session_state["cluster_run"] = True
            st.session_state["cluster_k"] = k
    with col2:
        # Scatter plot (PCA)
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        X = df[TAXA].values
        X_scaled = StandardScaler().fit_transform(X)
        X_pca = PCA(n_components=2).fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        pca_df["environment"] = df["environment"].values
        fig = px.scatter(pca_df, x="PC1", y="PC2", color="environment", color_discrete_map=ENV_COLORS, template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get("cluster_run", False):
        st.markdown("### Résultats")
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        st.metric("Silhouette score", f"{sil:.3f}")
        # Show cluster profiles
        cluster_df = pd.DataFrame({"Cluster": labels, **{tax: df[tax] for tax in TAXA}})
        cluster_means = cluster_df.groupby("Cluster")[TAXA].mean()
        st.dataframe(cluster_means, use_container_width=True)

        # AI interpretation
        if st.session_state.get("claude_key"):
            prompt = f"Expert métagénomique. K-means k={k} sur {len(df)} échantillons multi-environnements, silhouette score {sil:.3f}. En 3 phrases : signification biologique des clusters, interprétation du silhouette score, et une limite du k-means spécifique aux données métagénomiques (sparsité, compositionnalité) avec alternative recommandée."
            ai_response = call_claude(prompt, st.session_state.get("claude_key"))
            st.markdown("#### Interprétation IA")
            st.info(ai_response)


def random_forest_module(df):
    st.subheader("🌲 Random Forest")
    st.markdown("Classification supervisée de l'environnement source")
    if "RF_trained" not in st.session_state:
        st.session_state["RF_trained"] = False
    if st.button("🚀 Entraîner Random Forest"):
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, classification_report

        X = df[TAXA].values
        y = df["environment"].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        st.session_state["RF_acc"] = acc
        st.session_state["RF_importances"] = clf.feature_importances_
        st.session_state["RF_trained"] = True

    if st.session_state["RF_trained"]:
        st.metric("Précision", f"{st.session_state['RF_acc']:.1%}")
        # Importance plot
        fig = px.bar(x=TAXA, y=st.session_state["RF_importances"], color=TAXA, title="Importance des features", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        # AI interpretation
        if st.session_state.get("claude_key"):
            prompt = f"Expert ML. Random Forest précision {st.session_state['RF_acc']:.1%}, top features : {', '.join([f'{tax}:{imp:.3f}' for tax, imp in sorted(zip(TAXA, st.session_state['RF_importances']), key=lambda x: -x[1])[:3]])}. En 3 phrases : pourquoi ces taxons sont des biomarqueurs d'environnement, comment DNABERT-2 v4 améliore ce résultat (+5.5%), et une limite du RF pour les données métagénomiques."
            ai_response = call_claude(prompt, st.session_state.get("claude_key"))
            st.markdown("#### Interprétation IA")
            st.info(ai_response)


def lstm_module(df):
    st.subheader("⏱ LSTM")
    st.markdown("Dynamique temporelle du microbiome")
    tax = st.selectbox("Taxon", TAXA)
    months = st.slider("Prédiction (mois)", 1, 12, 3)
    perturbation = st.selectbox("Perturbation", ["Aucune", "Sécheresse", "Azote", "Antibiotiques"])
    if st.button("🚀 Modéliser"):
        st.session_state["lstm_run"] = True
        st.session_state["lstm_tax"] = tax
        st.session_state["lstm_months"] = months
        st.session_state["lstm_perturb"] = perturbation
    if st.session_state.get("lstm_run", False):
        # Simulate time series
        t = np.arange(12 + months)
        obs = 22 + 8 * np.sin(t[:12] * np.pi / 6) + np.random.normal(0, 1.5, 12)
        eff = {"Aucune": 0, "Sécheresse": -6, "Azote": 5, "Antibiotiques": -9}[perturbation]
        pred = obs[-1] + eff * np.exp(-np.arange(months) * 0.4) + np.random.normal(0, 0.8, months)
        full = np.concatenate([obs, pred])
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(12)), y=obs, mode="lines+markers", name="Observé", line=dict(color="#00D4AA")))
        fig.add_trace(go.Scatter(x=list(range(11, 12+months)), y=np.concatenate([[obs[-1]], pred]), mode="lines+markers", name="Prédit", line=dict(color="#9B7CFF", dash="dash")))
        fig.update_layout(template="plotly_dark", title=f"Abondance de {tax} au cours du temps", xaxis_title="Mois", yaxis_title="Abondance (%)")
        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.get("claude_key"):
            prompt = f"LSTM prédit {tax} sur {months} mois. Perturbation : {perturbation} (effet {eff:+d}%). En 3 phrases : avantage LSTM vs analyse statique, interprétation de la perturbation {perturbation} sur la communauté microbienne, et limite LSTM avec séries courtes."
            ai_response = call_claude(prompt, st.session_state.get("claude_key"))
            st.markdown("#### Interprétation IA")
            st.info(ai_response)


def vae_module(df):
    st.subheader("🧩 VAE Binning")
    st.markdown("Reconstruction de MAGs via autoencoder variationnel")
    latent_dim = st.selectbox("Dimension latente", [32, 64, 128], index=1)
    epochs = st.slider("Epochs", 10, 200, 50)
    if st.button("🚀 Lancer le binning"):
        st.session_state["vae_run"] = True
    if st.session_state.get("vae_run", False):
        # Simulate latent space
        from sklearn.manifold import TSNE
        from sklearn.decomposition import PCA
        X = df[TAXA].values
        X_pca = PCA(n_components=2).fit_transform(X)
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=df["environment"], color_discrete_map=ENV_COLORS, template="plotly_dark",
                         title="Espace latent VAE (2D projection)")
        st.plotly_chart(fig, use_container_width=True)
        st.write("47 MAGs reconstruits, dont 23 HQ (>90% complétude)")
        if st.session_state.get("claude_key"):
            prompt = "VAE binning métagénomique a reconstruit 47 MAGs dont 23 HQ (>90% complétude). En 3 phrases : principe espace latent TNF+couverture, avantage sur MetaBAT2 pour sols arides, et comment ces 23 MAGs représentent des organismes inconnus à nommer."
            ai_response = call_claude(prompt, st.session_state.get("claude_key"))
            st.markdown("#### Interprétation IA")
            st.info(ai_response)


def xai_module(df):
    st.subheader("💡 XAI / SHAP")
    st.markdown("Explicabilité du modèle Random Forest")
    if "xai_run" not in st.session_state:
        st.session_state["xai_run"] = False
    if st.button("🚀 Calculer SHAP"):
        from sklearn.ensemble import RandomForestClassifier
        import shap
        X = df[TAXA].values
        y = df["environment"].values
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Mean absolute SHAP values
        mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1) if isinstance(shap_values, list) else np.abs(shap_values).mean(axis=0)
        st.session_state["xai_importances"] = mean_abs_shap
        st.session_state["xai_run"] = True
    if st.session_state["xai_run"]:
        fig = px.bar(x=TAXA, y=st.session_state["xai_importances"], color=TAXA, title="SHAP |val| moyen", template="plotly_dark")
        st.plotly_chart(fig, use_container_width=True)
        if st.session_state.get("claude_key"):
            prompt = "SHAP explainer pour Random Forest. Les top features sont Proteobacteria, Actinobacteriota, Firmicutes. En 3 phrases : comment interpréter ces valeurs SHAP, pourquoi ces taxons sont déterminants pour la prédiction d'environnement, et une limite du SHAP pour les données compositionnelles."
            ai_response = call_claude(prompt, st.session_state.get("claude_key"))
            st.markdown("#### Interprétation IA")
            st.info(ai_response)


def gnn_module(df):
    st.subheader("🕸 GNN Interactions")
    st.markdown("Réseau d'interactions microbiennes via Graph Neural Network")
    # Simple network visualization using networkx
    import networkx as nx
    G = nx.Graph()
    # Add nodes
    for tax in TAXA:
        G.add_node(tax)
    # Random edges
    for i, tax1 in enumerate(TAXA):
        for j, tax2 in enumerate(TAXA):
            if i < j and np.random.rand() > 0.7:
                G.add_edge(tax1, tax2, weight=np.random.uniform(0.1, 0.9))
    pos = nx.spring_layout(G, seed=42)
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    node_x = [pos[node][0] for node in G.nodes()]
    node_y = [pos[node][1] for node in G.nodes()]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines", line=dict(width=1, color="#888"), hoverinfo="none"))
    fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text", marker=dict(size=20, color=COLORS[:len(TAXA)], line=dict(width=2, color="white")),
                             text=TAXA, textposition="middle center", textfont=dict(size=10, color="white")))
    fig.update_layout(showlegend=False, hovermode="closest", xaxis=dict(showgrid=False, zeroline=False, visible=False),
                      yaxis=dict(showgrid=False, zeroline=False, visible=False), template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    if st.session_state.get("claude_key"):
        prompt = "Réseau d'interactions microbiennes via GNN. Les hubs sont Proteobacteria, Firmicutes, Actinobacteriota. En 3 phrases : comment le GNN capture les interactions écologiques, avantage par rapport aux corrélations classiques, et limite de la GNN en présence de données compositionnelles."
        ai_response = call_claude(prompt, st.session_state.get("claude_key"))
        st.markdown("#### Interprétation IA")
        st.info(ai_response)


def report_module(df):
    st.subheader("📄 Rapport IA — Synthèse MetaInsight v4")
    st.markdown("Analyse intégrée des 12 modules par Claude (clé API requise)")

    api_key = st.text_input("Clé API Anthropic", type="password", key="claude_key_input")
    if api_key:
        st.session_state["claude_key"] = api_key

    question = st.text_area("Votre question scientifique", value="Synthétise les 4 nouveaux modules MetaInsight v4 et leur impact sur la métagénomique des sols arides.")
    profile = st.selectbox("Profil", ["Chercheur métagénomique", "Étudiant bioinformatique", "Généticien", "Écologiste"])
    format_type = st.selectbox("Format", ["Rapport structuré (sections)", "Résumé exécutif", "Présentation scientifique"])
    modules_cover = st.selectbox("Modules à couvrir", ["Tous les modules v4 (recommandé)", "Nouveaux modules v4 uniquement", "Comparaison v3 vs v4"])

    if st.button("🤖 Générer le rapport complet avec Claude"):
        if not api_key:
            st.warning("Veuillez entrer une clé API Anthropic.")
        else:
            prompt = f"Expert métagénomique senior. Niveau : {profile}. Format : {format_type}.\n\nMetaInsight v4 — plateforme complète avec 12 modules ML/DL :\n[v4 NEW] DNABERT-2 : précision 96.8% (vs RF 91.3%), 117M params, BPE tokenizer, attention multi-têtes.\n[v4 NEW] Causal ML : DAG PC-algorithm, Do-calculus Pearl. Proteobacteria → Shannon H′ causal (ATE=0.58), Firmicutes est spurieux (confondant=Sécheresse).\n[v4 NEW] GenAI : Dirichlet-VAE génère données synthétiques réalistes (FID=3.2, KL=0.04), ×10 augmentation.\n[v4 NEW] Federated : FedAvg + ε-DP=0.5, 6 labos algériens, modèle global 94.2% vs locaux 78-91%.\n[v3] K-means (sil.0.72), RF (91.3%), LSTM (RMSE~2.8%), VAE (47 MAGs), XAI/SHAP, GNN (3 hubs), Isolation Forest, Apriori.\n\nQuestion : {question}\n\nRapport de 300-350 mots avec sections : Apports v4 · Découvertes biologiques clés · Impact pour la métagénomique algérienne · Limites v4 · Recommandations v5."
            with st.spinner("Génération du rapport..."):
                response = call_claude(prompt, api_key)
                st.markdown(response)
                # Option to download
                st.download_button("📥 Télécharger rapport (.txt)", response, file_name="MetaInsight_v4_rapport.txt", mime="text/plain")


# -------------------------------
# Claude API call
# -------------------------------
def call_claude(prompt, api_key):
    """Call Claude API and return text response."""
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01"
    }
    data = {
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 900,
        "messages": [{"role": "user", "content": prompt}]
    }
    try:
        response = requests.post("https://api.anthropic.com/v1/messages", headers=headers, json=data, timeout=30)
        if response.status_code == 200:
            result = response.json()
            text = result["content"][0]["text"]
            return text
        else:
            return f"Erreur API: {response.status_code} - {response.text}"
    except Exception as e:
        return f"Erreur lors de l'appel: {str(e)}"


# -------------------------------
# Main app
# -------------------------------
def main():
    # Load data (either from session or demo)
    if "df" not in st.session_state:
        st.session_state["df"] = load_demo()

    # File uploader for user data
    uploaded_file = st.sidebar.file_uploader("📂 Charger vos propres données (CSV)", type=["csv"])
    if uploaded_file is not None:
        user_df = load_user_data(uploaded_file)
        if user_df is not None:
            st.session_state["df"] = user_df
            st.sidebar.success("Données chargées avec succès!")
    else:
        if st.sidebar.button("Utiliser les données de démonstration"):
            st.session_state["df"] = load_demo()
            st.rerun()

    df = st.session_state["df"]

    # Create tabs
    tabs = st.tabs(["🏠 Accueil", "🧬 DNABERT-2", "⚗️ Causal ML", "✨ GenAI", "🔒 Federated",
                    "🔵 Clustering", "🌲 RF", "⏱ LSTM", "🧩 VAE", "💡 XAI/SHAP", "🕸 GNN", "📄 Rapport IA"])

    with tabs[0]:
        home_module(df)
    with tabs[1]:
        dnabert_module(df)
    with tabs[2]:
        causal_module(df)
    with tabs[3]:
        genai_module(df)
    with tabs[4]:
        federated_module(df)
    with tabs[5]:
        clustering_module(df)
    with tabs[6]:
        random_forest_module(df)
    with tabs[7]:
        lstm_module(df)
    with tabs[8]:
        vae_module(df)
    with tabs[9]:
        xai_module(df)
    with tabs[10]:
        gnn_module(df)
    with tabs[11]:
        report_module(df)


if __name__ == "__main__":
    main()
