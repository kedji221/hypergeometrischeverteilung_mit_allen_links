import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import hypergeom, norm, binom
from scipy.special import comb
import pandas as pd

st.set_page_config(layout="wide")
st.title("Hypergeometrische Verteilung und Approximationen")

# Gemeinsame Sidebar-Parameter mit eindeutigen keys
with st.sidebar:
    st.header("Parameter")
    N = st.number_input("N (Größe der Grundgesamtheit)", min_value=1, max_value=10000, value=50, key="N")
    M = st.number_input("M (Anzahl der Erfolgselemente)", min_value=0, max_value=int(N), value=min(20, int(N)), key="M")
    n = st.number_input("n (Stichprobenumfang)", min_value=0, max_value=int(N), value=min(15, int(N)), key="n")
    k_default = max(0, min(M, n))
    k = st.number_input("k (Anzahl beobachteter Erfolge)", min_value=0, max_value=max(M,n), value=k_default, key="k")
    show_cc_area = st.checkbox("Stetigkeitskorrektur-Fläche im Diagramm zeigen", value=True, key="show_cc")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. Wahrscheinlichkeitsfunktion",
    "2. Kumulative Wahrscheinlichkeit",
    "3. Komplementäre kumulative Wahrscheinlichkeit",
    "4. Normalapproximation",
    "5. Binomialapproximation"
])

# --- Tab 1: PMF Wahrscheinlichkeitsfunktion ---
with tab1:
    st.title("Hypergeometrische Verteilung: Wahrscheinlichkeitsfunktion")
    k_min = max(0, n + M - N)
    k_max = min(M, n)
    if k < k_min: k = k_min
    if k > k_max: k = k_max

    bin1 = comb(M, k, exact=True)
    bin2 = comb(N - M, n - k, exact=True)
    bin3 = comb(N, n, exact=True)
    pmf_calc = bin1 * bin2 / bin3
    pmf_scipy = hypergeom.pmf(k, N, M, n)

    st.markdown(
        r"""
        **Formel der hypergeometrischen Verteilung:**

        $$
        P(X = k) = \frac{\dbinom{M}{k} \cdot \dbinom{N-M}{n-k}}{\dbinom{N}{n}}
        $$

        - $N$ = Größe der Grundgesamtheit   
        - $M$ = Anzahl der Erfolgselemente   
        - $n$ = Stichprobengröße   
        - $k$ = Zahl der Erfolge in der Stichprobe   
        """
    )

    st.subheader(f"Berechnung für k = {k}")
    st.latex(
        rf"""
        P(X = {k}) = \frac{{\dbinom{{{M}}}{{{k}}} \cdot \dbinom{{{N-M}}}{{{n-k}}}}}{{\dbinom{{{N}}}{{{n}}}}}
        = \frac{{{bin1} \cdot {bin2}}}{{{bin3}}}
        = {pmf_calc:.5f}
        """
    )
    st.info(f"Ergebnis: {pmf_scipy:.5f}")

    k_values = np.arange(k_min, k_max + 1)
    pmf_values = hypergeom.pmf(k_values, N, M, n)

    fig, ax = plt.subplots()
    ax.bar(k_values, pmf_values, color="skyblue")
    ax.axvline(x=k, color='orange', linestyle='--', label=f"aktuelles k={k}")
    ax.set_xlabel("k (Erfolge in Stichprobe)")
    ax.set_ylabel("P(X = k)")
    ax.set_title("Hypergeometrische Verteilung - Wahrscheinlichkeitsfunktion")
    ax.legend()
    st.pyplot(fig)

    with st.expander("Tabelle der Wahrscheinlichkeiten für alle k"):
        st.dataframe({"k": k_values, "P(X = k)": pmf_values})

# --- Tab 2: Kumulative Wahrscheinlichkeit ---
with tab2:
    st.title("Hypergeometrische Verteilung: Kumulative Wahrscheinlichkeit P(X ≤ k)")
    k_limit = min(M, n)
    if k > k_limit: k = k_limit
    k_values = np.arange(0, k + 1)
    pmf_values = hypergeom.pmf(k_values, N, M, n)

    st.subheader(f"Berechnung der Einzelwahrscheinlichkeiten für k = 0 bis {k}")
    for i, p in zip(k_values, pmf_values):
        st.markdown(f"**P(X = {i}) =** {p:.6f}")

    sum_pmf = np.sum(pmf_values)
    st.subheader(f"Kumulative Wahrscheinlichkeit: P(X ≤ {k})")
    st.markdown(f"$$ P(X \\leq {k}) = \\sum_{{i=0}}^{k} P(X = i) = {sum_pmf:.6f} $$")

    fig, ax = plt.subplots()
    ax.bar(k_values, pmf_values, color='skyblue')
    ax.set_xlabel("k")
    ax.set_ylabel("P(X = k)")
    ax.set_title(f"Einzelwahrscheinlichkeiten P(X = k) für k=0..{k}")
    st.pyplot(fig)

    cdf_values = hypergeom.cdf(k_values, N, M, n)
    fig2, ax2 = plt.subplots()
    ax2.plot(k_values, cdf_values, marker='o', linestyle='-', color='darkgreen')
    ax2.set_xlabel("k")
    ax2.set_ylabel("P(X \\leq k)")
    ax2.set_title(f"Kumulative Verteilungsfunktion für k=0..{k}")
    st.pyplot(fig2)

# --- Tab 3: Komplementäre kumulative Wahrscheinlichkeit ---
with tab3:
    st.title("Hypergeometrische Verteilung: Komplementäre kumulative Wahrscheinlichkeit P(X ≥ k)")

    k_min = 0
    k_max = min(M, n)
    if k > k_max:
        st.warning(f"Der Wert k = {k} ist größer als der maximal mögliche Wert {k_max}. Bitte anpassen.")
    else:
        ks = np.arange(k, k_max + 1)
        pmf_vals = hypergeom.pmf(ks, N, M, n)
        sum_pmf = pmf_vals.sum()

        if k == 0:
            cdf_part = 0.0
        else:
            cdf_part = hypergeom.cdf(k - 1, N, M, n)
        complement_cdf_calc = 1 - cdf_part

        st.subheader(f"Einzelwahrscheinlichkeiten für k = {k} bis {k_max}")
        for i, val in zip(ks, pmf_vals):
            st.markdown(f"**P(X = {i}) =** {val:.6f}")

        st.subheader(f"Berechnung der komplementären kumulativen Wahrscheinlichkeit P(X ≥ {k})")
        st.markdown(
            rf"""
            $$ P(X \geq {k}) = \sum_{{i={k}}}^{{{k_max}}} P(X = i) = {sum_pmf:.6f} $$
            
            $$ P(X \geq {k}) = 1 - P(X \leq {k-1}) = 1 - {cdf_part:.6f} = {complement_cdf_calc:.6f} $$
            """
        )

        fig1, ax1 = plt.subplots()
        ax1.bar(ks, pmf_vals, color="lightcoral")
        ax1.set_xlabel("k (Erfolge in Stichprobe)")
        ax1.set_ylabel("P(X = k)")
        ax1.set_title(f"Einzelwahrscheinlichkeiten P(X = k) für k={k} bis {k_max}")
        st.pyplot(fig1)

        all_ks = np.arange(k_min, k_max + 1)
        all_cdf = hypergeom.cdf(all_ks - 1, N, M, n)
        comp_cdf_vals = 1 - all_cdf

        fig2, ax2 = plt.subplots()
        ax2.plot(all_ks, comp_cdf_vals, marker='o', linestyle='-', color='darkred', label="P(X ≥ k)")
        ax2.axvline(k, color='orange', linestyle='--', label=f"aktuelles k={k}")
        ax2.set_xlabel("k")
        ax2.set_ylabel("P(X ≥ k)")
        ax2.set_title("Komplementäre kumulative Verteilungsfunktion P(X ≥ k)")
        ax2.legend()
        st.pyplot(fig2)

# --- Tab 4: Normalapproximation ---
with tab4:
    st.title("Normalapproximation der Hypergeometrischen Verteilung")

    k_min = int(max(0, n + M - N))
    k_max = int(min(M, n))
    if k < k_min: k = k_min
    if k > k_max: k = k_max

    k_values = np.arange(k_min, k_max + 1)
    mu = n * M / N
    var = n * M * (N - M) * (N - n) / (N ** 2 * (N - 1))
    sigma = np.sqrt(var)

    st.markdown("### N(μ; σ)")
    st.latex(r"\mu = n \cdot \frac{M}{N}")
    st.write(f"μ = {n} × {M}/{N} = {mu:.3f}")

    st.latex(r"\sigma^2 = n \cdot \frac{M}{N} \cdot \left(1 - \frac{M}{N}\right) \cdot \frac{N - n}{N - 1}")
    st.write(f"σ² = {var:.3f}")

    st.latex(r"\sigma = \sqrt{\sigma^2}")
    st.write(f"σ = √{var:.3f} = {sigma:.3f}")

    with st.expander("Details zur Stetigkeitskorrektur", expanded=True):
        phi_plus = norm.cdf(k + 0.5, mu, sigma)
        phi_minus = norm.cdf(k - 0.5, mu, sigma)
        prob_eq = phi_plus - phi_minus
        prob_le = phi_plus
        prob_ge = 1 - phi_minus

        st.markdown(
            f"""
            Die **Stetigkeitskorrektur** ist notwendig, da die Hypergeometrische Verteilung diskret ist, 
            während die Normalverteilung stetig ist.

            #### Berechnungen für k = {k} mit Stetigkeitskorrektur:
            - **P(X = {k}) ≈ Φ({k}+0.5) - Φ({k}-0.5)**   
              = Φ({k}+0.5) = {phi_plus:.5f}   
              − Φ({k}-0.5) = {phi_minus:.5f}   
              ⇒ **P(X = {k}) ≈ {prob_eq:.5f}**

            - **P(X ≤ {k}) ≈ Φ({k}+0.5)**   
              ⇒ **P(X ≤ {k}) ≈ {prob_le:.5f}**

            - **P(X ≥ {k}) ≈ 1 − Φ({k}-0.5)**   
              ⇒ **P(X ≥ {k}) ≈ {prob_ge:.5f}**

            *Φ ist die Verteilungsfunktion der Standardnormalverteilung.*
            """,
            unsafe_allow_html=True
        )

    pmf_values = hypergeom.pmf(k_values, N, M, n)
    normal_pmf_approx = norm.cdf(k_values + 0.5, mu, sigma) - norm.cdf(k_values - 0.5, mu, sigma)
    cdf_values = hypergeom.cdf(k_values, N, M, n)
    ccdf_values = 1 - hypergeom.cdf(k_values - 1, N, M, n)
    normal_cdf_approx = norm.cdf(k_values + 0.5, mu, sigma)
    normal_ccdf_approx = 1 - norm.cdf(k_values - 0.5, mu, sigma)

    bin1 = comb(M, k, exact=True)
    bin2 = comb(N - M, n - k, exact=True)
    bin3 = comb(N, n, exact=True)
    pmf_calc = bin1 * bin2 / bin3
    pmf_hygeo = hypergeom.pmf(k, N, M, n)
    pmf_norm = norm.cdf(k + 0.5, mu, sigma) - norm.cdf(k - 0.5, mu, sigma)
    cdf_hygeo = hypergeom.cdf(k, N, M, n)
    cdf_norm = norm.cdf(k + 0.5, mu, sigma)
    ccdf_hygeo = 1 - hypergeom.cdf(k - 1, N, M, n)
    ccdf_norm = 1 - norm.cdf(k - 0.5, mu, sigma)

    st.subheader(f"Berechnung für k = {k}")
    st.latex(
        rf"""
        P(X = {k}) = \frac{{\dbinom{{{M}}}{{{k}}} \cdot \dbinom{{{N-M}}}{{{n-k}}}}}{{\dbinom{{{N}}}{{{n}}}}}
        = \frac{{{bin1} \cdot {bin2}}}{{{bin3}}}
        = {pmf_calc:.5f}
        """
    )
    st.info(f"P(X = {k}) (hypergeometrisch) = {pmf_hygeo:.5f} | (Normalapprox. mit Korrektur) ≈ {pmf_norm:.5f}")

    st.latex(
        rf"""
        P(X \leq {k}) = \sum_{{i={k_min}}}^{{{k}}} P(X = i)
        """
    )
    st.info(f"P(X ≤ {k}) (hypergeometrisch) = {cdf_hygeo:.5f} | (Normalapprox. mit Korrektur) ≈ {cdf_norm:.5f}")

    st.latex(
        rf"""
        P(X \geq {k}) = 1 - P(X < {k}) = 1 - P(X \leq {k-1})
        """
    )
    st.info(f"P(X ≥ {k}) (hypergeometrisch) = {ccdf_hygeo:.5f} | (Normalapprox. mit Korrektur) ≈ {ccdf_norm:.5f}")

    st.subheader("1. Dichtefunktion P(X = k)")
    col_pmf_plot, col_pmf_legende = st.columns([3, 1])
    with col_pmf_plot:
        fig_pmf, ax_pmf = plt.subplots()
        ax_pmf.bar(k_values, pmf_values, color="skyblue", label="PMF: diskret (Hypergeometrisch)")
        ax_pmf.plot(
            k_values, normal_pmf_approx, color="crimson", marker="o", linestyle="dashed",
            label="Normalapproximation (mit Kontinuitätskorrektur)"
        )
        ax_pmf.axvline(x=k, color='orange', linestyle='--', label=f"aktuelles k={k}")

        if show_cc_area:
            x_fill = np.linspace(k - 0.5, k + 0.5, 250)
            y_norm = norm.pdf(x_fill, mu, sigma)
            ax_pmf.fill_between(x_fill, y_norm, color="gold", alpha=0.5, label="Stetigkeitskorrektur-Fläche")
            ax_pmf.plot([k - 0.5, k - 0.5], [0, norm.pdf(k - 0.5, mu, sigma)], color="gold", lw=2, alpha=0.7)
            ax_pmf.plot([k + 0.5, k + 0.5], [0, norm.pdf(k + 0.5, mu, sigma)], color="gold", lw=2, alpha=0.7)

        ax_pmf.set_xlabel("k (Erfolge in Stichprobe)")
        ax_pmf.set_ylabel("P(X = k)")
        ax_pmf.set_title("Dichtefunktion: Hypergeometrisch & Normalapproximation")
        # HIER: Legende NICHT anzeigen!
        # ax_pmf.legend()  <-- Entfernt!
        st.pyplot(fig_pmf)

    with col_pmf_legende:
        st.markdown(
            """
            <br>
            <b>Legende:</b>
            <ul>
                <li><span style='color:skyblue; font-size:1.5em;'>■</span> PMF: diskret (Hypergeometrisch)</li>
                <li><span style='color:crimson; font-size:1.5em;'>●</span> Normalapproximation (mit Kontinuitätskorrektur)</li>
                <li><span style='color:gold; font-size:1.5em;'>■</span> Stetigkeitskorrektur-Fläche</li>
                <li><span style='color:orange; font-size:1.5em;'>│</span> aktuelles k</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    st.subheader("Verteilungsfunktion P(X ≤ k) und Komplementärfunktion P(X ≥ k) mit Normalapproximation")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Verteilungsfunktion: $P(X \\leq k)$**")
        fig_cdf, ax_cdf = plt.subplots()
        ax_cdf.step(k_values, cdf_values, where="post", color="navy", label="hypergeom (diskret)")
        ax_cdf.plot(k_values, normal_cdf_approx, color="crimson", linestyle="--", label="Normalapprox. (mit Korrektur)")
        ax_cdf.axvline(x=k, color='orange', linestyle='--', label=f"aktuelles k={k}")
        ax_cdf.set_xlabel("k (Erfolge)")
        ax_cdf.set_ylabel("P(X ≤ k)")
        ax_cdf.set_title("P(X ≤ k): Hypergeometrisch & Normalapproximation")
        ax_cdf.legend()
        st.pyplot(fig_cdf)

    with col2:
        st.markdown("**Komplementäre Funktion: $P(X \\geq k)$**")
        fig_ccdf, ax_ccdf = plt.subplots()
        ax_ccdf.step(k_values, ccdf_values, where="post", color="green", label="hypergeom (diskret)")
        ax_ccdf.plot(k_values, normal_ccdf_approx, color="crimson", linestyle="--", label="Normalapprox. (mit Korrektur)")
        ax_ccdf.axvline(x=k, color='orange', linestyle='--', label=f"aktuelles k={k}")
        ax_ccdf.set_xlabel("k (Erfolge)")
        ax_ccdf.set_ylabel("P(X ≥ k)")
        ax_ccdf.set_title("P(X ≥ k): Hypergeometrisch & Normalapproximation")
        ax_ccdf.legend()
        st.pyplot(fig_ccdf)

    df = pd.DataFrame({
        "k": k_values,
        "P(X = k)": np.round(pmf_values, 6),
        "Norm. appr. P(X = k)": np.round(normal_pmf_approx, 6),
        "P(X ≤ k)": np.round(cdf_values, 6),
        "Norm. appr. P(X ≤ k)": np.round(normal_cdf_approx, 6),
        "P(X ≥ k)": np.round(ccdf_values, 6),
        "Norm. appr. (X ≥ k)": np.round(normal_ccdf_approx, 6),
    })
    st.subheader("Tabelle: Diskrete Werte & Normalapproximation (mit Kontinuitätskorrektur)")
    st.dataframe(df, hide_index=True)

# --- Tab 5: Binomialapproximation ---
with tab5:
    st.title("Binomialapproximation der Hypergeometrischen Verteilung")

    k_min = int(max(0, n + M - N))
    k_max = int(min(M, n))
    if k < k_min: k = k_min
    if k > k_max: k = k_max

    k_values = np.arange(k_min, k_max + 1)
    p = M / N

    pmf_values = hypergeom.pmf(k_values, N, M, n)
    cdf_values = hypergeom.cdf(k_values, N, M, n)
    ccdf_values = 1 - hypergeom.cdf(k_values - 1, N, M, n)

    binom_pmf = binom.pmf(k_values, n, p)
    binom_cdf = binom.cdf(k_values, n, p)
    binom_ccdf = 1 - binom.cdf(k_values - 1, n, p)

    bin1 = comb(M, k, exact=True)
    bin2 = comb(N - M, n - k, exact=True)
    bin3 = comb(N, n, exact=True)
    pmf_calc = bin1 * bin2 / bin3
    pmf_hygeo = hypergeom.pmf(k, N, M, n)
    pmf_binom = binom.pmf(k, n, p)
    cdf_hygeo = hypergeom.cdf(k, N, M, n)
    cdf_binom = binom.cdf(k, n, p)
    ccdf_hygeo = 1 - hypergeom.cdf(k - 1, N, M, n)
    ccdf_binom = 1 - binom.cdf(k - 1, n, p)

    st.subheader(f"Berechnung für k = {k}")
    st.latex(
        rf"""
        P(X = {k}) = \frac{{\dbinom{{{M}}}{{{k}}} \cdot \dbinom{{{N-M}}}{{{n-k}}}}}{{\dbinom{{{N}}}{{{n}}}}}
        = \frac{{{bin1} \cdot {bin2}}}{{{bin3}}}
        = {pmf_calc:.5f}
        """
    )
    st.info(f"P(X = {k}) (hypergeometrisch) = {pmf_hygeo:.5f} | (Binomialapprox.) = {pmf_binom:.5f}")

    st.latex(rf"P(X \leq {k}) = \sum_{{i={k_min}}}^{{{k}}} P(X = i)")
    st.info(f"P(X ≤ {k}) (hypergeometrisch) = {cdf_hygeo:.5f} | (Binomialapprox.) = {cdf_binom:.5f}")

    st.latex(rf"P(X \geq {k}) = 1 - P(X < {k}) = 1 - P(X \leq {k-1})")
    st.info(f"P(X ≥ {k}) (hypergeometrisch) = {ccdf_hygeo:.5f} | (Binomialapprox.) = {ccdf_binom:.5f}")

    col_pmf_plot, col_pmf_legende = st.columns([3, 1])
    with col_pmf_plot:
        fig_pmf, ax_pmf = plt.subplots()
        ax_pmf.bar(k_values, pmf_values, color="skyblue", label="PMF: diskret (Hypergeometrisch)")
        ax_pmf.bar(k_values, binom_pmf, color="darkred", alpha=0.7, label="Dichte: Binomialverteilung")
        ax_pmf.axvline(x=k, color='orange', linestyle='--', label=f"aktuelles k={k}")

        ax_pmf.set_xlabel("k (Erfolge in Stichprobe)")
        ax_pmf.set_ylabel("P(X = k)")
        ax_pmf.set_title("Dichtefunktion: Hypergeometrisch & Binomialverteilung")
        st.pyplot(fig_pmf)

    with col_pmf_legende:
        st.markdown(
            """
            <br>
            <b>Legende:</b>
            <ul>
                <li><span style='color:skyblue; font-size:1.5em;'>■</span> Hypergeometrische Verteilung</li>
                <li><span style='color:darkred; font-size:1.5em;'>■</span> Binomialverteilung</li>
                <li><span style='color:orange; font-size:1.5em;'>│</span> aktuelles k</li>
            </ul>
            """,
            unsafe_allow_html=True,
        )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Verteilungsfunktion: $P(X \\leq k)$**")
        fig_cdf, ax_cdf = plt.subplots()
        ax_cdf.plot(k_values, cdf_values, color="navy", linestyle='-', marker='o', label="hypergeom")
        ax_cdf.plot(k_values, binom_cdf, color="darkred", linestyle="--", label="Binomialverteilung")
        ax_cdf.axvline(x=k, color='orange', linestyle='--', label=f"aktuelles k={k}")
        ax_cdf.set_xlabel("k (Erfolge)")
        ax_cdf.set_ylabel("P(X ≤ k)")
        ax_cdf.set_title("P(X ≤ k): Hypergeometrisch & Binomialverteilung")
        ax_cdf.legend(loc='best')
        st.pyplot(fig_cdf)

    with col2:
        st.markdown("**Komplementäre Funktion: $P(X \\geq k)$**")
        fig_ccdf, ax_ccdf = plt.subplots()
        ax_ccdf.plot(k_values, ccdf_values, color="green", linestyle='-', marker='o', label="hypergeom")
        ax_ccdf.plot(k_values, binom_ccdf, color="darkred", linestyle="--", label="Binomialverteilung")
        ax_ccdf.axvline(x=k, color='orange', linestyle='--', label=f"aktuelles k={k}")
        ax_ccdf.set_xlabel("k (Erfolge)")
        ax_ccdf.set_ylabel("P(X ≥ k)")
        ax_ccdf.set_title("P(X ≥ k): Hypergeometrisch & Binomialverteilung")
        ax_ccdf.legend(loc='best')
        st.pyplot(fig_ccdf)

    df = pd.DataFrame({
        "k": k_values,
        "P(X = k)": np.round(pmf_values, 6),
        "Binomial P(X = k)": np.round(binom_pmf, 6),
        "P(X ≤ k)": np.round(cdf_values, 6),
        "Binomial (X ≤ k)": np.round(binom_cdf, 6),
        "P(X ≥ k)": np.round(ccdf_values, 6),
        "Binomial (X ≥ k)": np.round(binom_ccdf, 6),
    })
    st.subheader("Tabelle: Hypergeometrische Verteilung und Binomialapproximation")
    st.dataframe(df, hide_index=True)
