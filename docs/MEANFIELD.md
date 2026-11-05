# Mean-Field-Signaturen der Winkel zwischen Werterichtungen

Code: `valueheads_pythia/analysis/meanfield.py` · Tests: `tests/test_meanfield.py` · Notebook:
`notebooks/runs/meanfield-angles.ipynb` (erzeugt von `scripts/make_meanfield_notebook.py`).

## Frage

Das Wert-Head-Paper (arXiv 2611.02503, App. K, Tab. 17) misst die Winkel zwischen den Richtungsvektoren zweier Werte auf
den Heads, die in beiden Top-15 %-Mengen liegen. Für Qwen3-4B: harmless × honesty 88.6° ± 9.7, helpful × honesty
79.0° ± 10.0, harmless × helpful 69.1° ± 12.6, dazu „eine kleine Teilmenge von Heads mit Richtungskonflikt (90–120°)“.

These: Verwandte Werte sind „nur eine 90-Grad-Drehung“; interessant ist das Residuum, das nicht passt. Mittelt sich dieses
Residuum über die Heads heraus, liegen die Werte im „Mean-Field-Regime“, sonst in einem anderen Regime.

Wichtig vorab: Zufällige Einheitsvektoren in R^d stehen fast senkrecht aufeinander. 90° ist also der *Normalfall*, kein
Befund. Aussagekräftig ist nur der **Überschuss** über die isotrope Null (N1) und über das untrainierte Modell (step0, N2).

## Was gemessen wird

Für jedes Modell, jeden Checkpoint, jedes Wertepaar (harmless × helpful, harmless × honesty, helpful × honesty) und zwei
Head-Sätze:

1. **geteilte Top-15 %-Heads** (wie Tab. 17; Top-15 % je Wert und Checkpoint aus der gespeicherten Rangfolge der
   Head-Tabelle, `probing.top_heads_from_file`, k = int(0.15 · L · H)),
2. **alle Heads** (L · H),

wird je Head h das **Residuum**

    r(h) = cos θ(h) = <dir_a[h], dir_b[h]> / (|dir_a[h]| · |dir_b[h]|),   θ(h) = arccos r(h)

berechnet. r = 0 heißt genau 90°, r > 0 ein gemeinsamer Anteil (Winkel < 90°), r < 0 ein Richtungskonflikt (> 90°).

Daten: `results/directions/{modell}_{checkpoint}_{wert}.npz` (`directions.save_directions`; gespeichert ist die **rohe**
Massenmittel-Differenz nach Gl. 2, nicht normiert – r ist skalenfrei), Form (L · H, D), Index = Layer · H + Head wie
`flat_idx` der Head-Tabellen `results/heads/{modell}_{checkpoint}_{wert}_pythia.json`. Die Geometrie wird gegen
`config.MODELS` geprüft: 160m 12 × 12 × 64, 410m 24 × 16 × 64, 1b 16 × 8 × 256, 1.4b 24 × 16 × 128, 2.8b 32 × 32 × 80.
Die Winkelstatistik im Stil von Tab. 17 (n, Mittel, Median, Populations-sd) wird mit `analysis/direction_angles.py`
(`angle_table`, `all_head_angle_table`) gegengerechnet; bei Abweichung bricht die Analyse ab.

HelpSteer-Werte liegen nur an einigen Checkpoints vor und sind in v1 ausgeschlossen.

## Nullmodelle

**N1 – isotrop.** Für zwei unabhängige, gleichverteilte Einheitsvektoren in R^d hat r die Dichte
∝ (1 − r²)^((d−3)/2) auf [−1, 1], d. h. (r + 1)/2 ~ Beta((d−1)/2, (d−1)/2): Mittel 0, Varianz 1/d, Exzess-Kurtosis
−6/(d+2). Quantile werden exakt berechnet (`beta.ppf`), nicht über die Normalnäherung.

| d | Modelle | Winkel-sd | exaktes 95 %-Intervall | Anteil > 100° |
|---|---|---|---|---|
| 64 | pythia-160m, pythia-410m | 7.2° | [75.9°, 104.1°] | 8.3 % |
| 80 | pythia-2.8b | 6.4° | [77.4°, 102.6°] | 6.1 % |
| 128 | pythia-1.4b (Qwen3-4B) | 5.1° | [80.0°, 100.0°] | 2.4 % |
| 256 | pythia-1b | 3.6° | [83.0°, 97.0°] | 0.26 % |

Die sd-Werte von Tab. 17 (9.7–12.6° bei d = 128) liegen also beim 1.9- bis 2.5-Fachen der isotropen Streuung.

**N2 – step0.** Dieselbe Messung im untrainierten Modell, **auf denselben Heads**. N2 fängt ab, was N1 nicht kennt: die
Anisotropie des Aktivierungsraums (zwei unabhängige Schätzungen in einem anisotropen Raum haben eine größere
cos-Streuung als 1/d, nämlich etwa 1/d_eff mit d_eff = tr(Σ)²/tr(Σ²)) und Daten-Artefakte, die schon ohne Training eine
gemeinsame Richtung erzeugen (harmless und helpful stammen beide aus HH-RLHF, gleiches Dialogformat).

**Keine Label-Permutations-Null.** Die Feature-Shards sind gelöscht; Richtungen unter permutierten Labels lassen sich nicht
mehr berechnen. Das Schätzrauschen der Massenmittel-Richtung ist ein zufälliger Vektor, der die geschätzten Richtungen
dekorreliert und Winkel Richtung 90° zieht. Eine gemessene Überschuss-Streuung (und eine Lage ≠ 0) ist deshalb
**konservativ**: Die wahren Residuen sind eher größer.

## Signaturen

Alle Konfidenzintervalle stammen aus 2000 Bootstrap-Ziehungen über Heads, alle Permutationstests aus 2000 Permutationen
(Seeds fest je Modell × Checkpoint × Paar × Head-Satz, Ergebnisse reproduzierbar).

**(a) Lage.** Mittleres r gegen 0 (N1) und mittlere Differenz r_t(h) − r_step0(h) auf denselben Heads (N2). KI und
p-Wert aus einem **Cluster-Bootstrap über Layer** (die Heads einer Schicht bleiben zusammen, damit korrelierte Heads die
Unsicherheit nicht unterschätzen; bei < 3 Layern über einzelne Heads), t-Test mit dem Bootstrap-Standardfehler
(Faktor G/(G−1) für G Cluster) und G − 1 Freiheitsgraden. Zusätzlich das zentrale 95 %-Intervall des Mittels unter N1
(±1.96/√(n d)). „Konsistentes Vorzeichen“: Mittel und Median haben dasselbe Vorzeichen.

**(b) Überschuss-Streuung.** var(r) − 1/d und sd(r)/sd_N1 = √(var(r) · d) mit KI, χ²-Test gegen N1 (einseitig,
Überschuss); Verhältnis sd(r_t)/sd(r_step0) auf denselben Heads (gepaarter Bootstrap). Das KI von var(r) (und damit von
var(r) − 1/d und sd(r)/sd_N1) ist die Vereinigung aus dem Perzentil-Bootstrap über Heads und dem χ²-Intervall der
Normaltheorie, (n − 1) s²/χ²_{n−1}: Der Bootstrap allein überdeckt in isotropen Simulationen den wahren Wert 1 nur in
93–95 % (alle Heads) bzw. 82–89 % (18–25 Heads) der Läufe, die Vereinigung in 94–98 %. Unter N1 ist r leicht
platykurtisch, das χ²-Intervall also exakt bis konservativ; bei schweren Rändern übernimmt die Bootstrap-Seite. Anteil der Heads
mit Winkel > 100° gegen seinen exakten N1-Erwartungswert (Binomialtest, einseitig) und gegen step0.

**(c) Mittelungsgesetz.** Unabhängige Residuen mitteln sich wie 1/√n heraus. Gemessen wird die sd des Mittels von r
über **Fenster aus n benachbarten Heads in Layer-Reihenfolge** (alle überlappenden Fenster, n auf einem log-Gitter
1 … N/4, höchstens 8 Werte) und die Steigung von log sd gegen log n (gewichtete Regression, Gewicht = Zahl unabhängiger
Fenster − 1). Unabhängige Residuen → −1/2; korrelierte (benachbarte Heads oder Layer ähnlich, Tiefentrend) → flacher.

*Was (c) nicht sieht:* Das Mittelungsgesetz erkennt nur Korrelation, die an die **Layer-Reihenfolge** gebunden ist
(Nachbar-Heads, Layer, Tiefentrend). Ein Anteil, den **alle** Heads gemeinsam haben (Äquikorrelation,
r(h) = c + e(h)), ist innerhalb einer Realisierung – eines Modells an einem Checkpoint – eine konstante Verschiebung: Die
Varianz der Fenstermittel ändert sich nicht, die Steigung bleibt −1/2. Ein solcher Anteil zeigt sich **nur als
Lageverschiebung** und wird allein von (a) erfasst; ist die Verschiebung klein, kann die Zeile trotzdem
„mean-field-artig“ heißen. Simulation (24 × 16 Heads, d = 128, r = c + √(1 − ρ) · e mit c ~ N(0, ρ/d) für alle Heads,
60 Läufe je ρ = 0.02 / 0.1 / 0.3): mittlere Steigung −0.50, KI-Überdeckung von −0.5 0.90–1.00, Permutations-p < 0.05
in 3–8 % der Läufe; (a) erkennt die Verschiebung in 27 / 65 / 85 % der Läufe, die Zeile heißt in 67 / 33 / 15 %
„mean-field-artig“. Ebenso blind ist (c) für einen Rest, der von Eigenschaften der Heads abhängt, die nicht an die
Layer-Reihenfolge gebunden sind (Probe-Score, Norm der Massenmittel-Differenz). Ein solcher Rest schrumpft genau wie
Rauschen mit 1/√n; ihn erkennt nur die Vorhersagbarkeit (g).

*Warum keine zufälligen Teilmengen allein:* Für **jede** feste Menge von N Werten gilt für zufällige Teilmengen ohne
Zurücklegen exakt sd(Mittel) = S · √((1 − n/N)/n). Nach der Endlichkeitskorrektur ist die Steigung also per Konstruktion
−1/2 – unabhängig davon, ob die Residuen korreliert sind (`random_subset_sd`, im Test belegt). Zufällige Teilmengen sind
deshalb die **Null** (dieselben Fenster auf permutierten Heads): Die rohe Steigung wird um den Null-Mittelwert
bias-korrigiert (der Logarithmus einer Varianzschätzung ist bei wenigen Fenstern verzerrt), sodass unabhängige Residuen
−0.5 ergeben. KI: Layer-Block-Bootstrap (ganze Layer mit Zurücklegen aneinandergereiht, auf N gekürzt), um die korrigierte
Steigung zentriert; jede Hälfte wird auf mindestens 1.96 × die sd der Permutationsnull verbreitert (unter Unabhängigkeit
ist das die Stichproben-sd der Steigung). Gemessen an 1000 isotropen Simulationen je Geometrie (B = P = 500) überdeckt
der Bootstrap allein −0.5 nur in 91–95 % der Läufe (12 × 12, d = 64: 0.92; 16 × 8, d = 256: 0.91; 24 × 16: 0.94;
32 × 32: 0.95), das verbreiterte KI in 95–97 %; die korrigierte Steigung ist im Mittel −0.500. Typische Breiten: alle
Heads 0.10–0.26, 40 Heads ≈ 0.45, 18–21 Heads 0.65–0.71. p: einseitiger Permutationstest (flacher als die Null). Dazu
n_eff = var(r)/var(Fenstermittel) beim größten Fenster mit ≥ 16 unabhängigen Fenstern und dessen Verhältnis zur Null
(1 = unabhängig, < 1 = weniger unabhängige Heads als Heads). Unter 16 Heads wird keine Steigung berechnet.

**(d) Korrelationsstruktur.** Einfache Varianzkomponenten von r nach Layer: ICC(1) (Korrelation innerhalb eines Layers),
Varianz zwischen und innerhalb der Layer, F. p aus einer Permutation der Layer-Etiketten (Gruppengrößen bleiben); der
F-Test-p steht daneben. Mindestens 8 Freiheitsgrade innerhalb der Layer und 3 Layer mit ≥ 2 Heads, sonst kein Test
(dünne Head-Sätze mit meist einem Head je Layer). Zwischen-Layer-Struktur deskriptiv: Lag-1-Autokorrelation der
Layer-Mittel gegen permutierte Layer-Reihenfolge.

**(e) Persistenz.** Spearman von r(h) zum vorherigen Checkpoint und zu step143000 auf denselben Heads; Null =
Permutation der Heads (einseitig, ρ > 0).

**(f) Form.** Exzess-Kurtosis mit KI (N1: −6/(d+2)). Bimodalität über ein Gaußsches Mischmodell mit 2 Komponenten gegen
die **bessere eingipflige Alternative** aus einer Normalverteilung (2 Parameter) und einer **Schiefnormalverteilung**
(3 Parameter, Maximum-Likelihood mit mehreren Startwerten): „bimodal“, wenn ΔBIC = min(BIC_Normal, BIC_Schiefnormal) −
BIC₂ > 10 (`gmm_dbic_skew`), Ashmans D > 2 (saubere Trennung) und die kleinere Komponente ≥ 5 % der Heads umfasst (ein
einzelner Ausreißer-Head ist keine Population); die Winkel beider Komponenten werden berichtet, so wird eine
„Richtungskonflikt“-Teilpopulation (z. B. 15 % bei 110°) sichtbar. `gmm_dbic` (nur gegen eine Normalverteilung) und
`gmm_modes` (Zahl der Gipfel der angepassten Mischdichte) stehen deskriptiv daneben.

*Warum die Schiefnormal-Alternative:* Gegen eine einzige Normalverteilung zerlegt das Mischmodell eine schiefe, aber
eingipflige Verteilung (für harmless × helpful plausibel) in einen schmalen Kern und einen breiten Rand mit D knapp über
2. Simulationen (100 Läufe je Zelle, n = 60 / 144 / 384, dazu 40 Läufe bei n = 1024): Die alte Regel nannte
Schiefnormal(a = 8) in 5 / 37 / 53 / 50 % der Läufe bimodal, Gamma(k = 2) in 18 / 29 / 16 / 3 %, Lognormal(s = 0.6) in
32 / 21 / 7 % (n ≤ 384); die neue Regel in höchstens 1 %. t₅ und isotrope r (n = 1024): 0 % (beide Regeln). Der Preis ist Power bei
kleinem n und kleinem d. 15 % der Heads bei 110° (Rest isotrop) gelten als bimodal:

| d | n = 60 | n = 144 | n = 384 | n = 1024 |
|---|---|---|---|---|
| 128, alte Regel → neue | 0.58 → 0.33 | 0.98 → 0.85 | 1.00 → 1.00 | – |
| 64, alte Regel → neue | 0.07 → 0.05 | 0.41 → 0.34 | 0.83 → 0.54 | 1.00 → 0.97 |

„Zwei Gipfel der angepassten Mischdichte“ allein wäre keine bessere Lösung: Bei d = 64 hat die Mischdichte des wahren
110°-Falls auch bei n = 1024 nur in 35 % der Läufe zwei Gipfel. Hartigans Dip-Test wird zusätzlich ausgegeben, wenn
das optionale Paket `diptest` installiert ist, geht aber nicht ins Urteil ein (damit das Urteil nicht von der Umgebung
abhängt; der Dip-Test hat zudem wenig Power für eine 15 %-Nebenkomponente).

## (g) Vorhersagbarkeit des Rests

*Warum:* Das 1/√n-Schrumpfen gemittelter Residuen – die Grundlage von (c) – unterscheidet reines Rauschen **nicht** von
einem strukturierten, systematischen Rest. Hängt r(h) von einer Eigenschaft des Heads ab, die nicht an die
Layer-Reihenfolge gebunden ist (etwa vom Probe-Score), dann sind die r(h) entlang der Layer-Reihenfolge so unabhängig wie
Rauschen und schrumpfen im Mittel genauso. Lage, Steigung, Layer-Test und Form sehen dann alle „mean-field-artig“ aus.
Erst die **Vorhersagbarkeit außerhalb der Stichprobe** trennt die beiden Fälle. Demo (24 × 16 Heads, d = 128,
60 Läufe je Fall, B = P = 500): Für unabhängiges Rauschen ist die Mittelungssteigung −0.504, ihr KI enthält −1/2 in
98 % der Läufe, das Held-out-R² beträgt −0.003. Für r = (0.9 · z(Probe-Score) + 0.44 · Rauschen)/√d sind es −0.503,
97 % und **+0.80**. Ohne (g) hieß der strukturierte Fall in 97 % der Läufe „mean-field-artig“, mit (g) in 0 %
(Test: `test_structured_residual_averages_like_noise_but_is_predictable`).

*Was:* Für jede Zeile (Modell × Checkpoint × Paar × Head-Satz) werden zwei Ziele vorhergesagt: r(h) mit Vorzeichen
(`pred_r_*`) und der Betrag |r(h)| (`pred_absr_*`). Auch ein Betrag, der von Head-Eigenschaften abhängt
(heteroskedastischer Rest), widerspricht dem Bild unabhängigen, gleichartigen Rauschens.

*Kovariaten je Head* (alle ohnehin vorhanden):

| Name | Bedeutung |
|---|---|
| `depth`, `depth2` | relative Tiefe Layer/(L − 1) und ihr zentriertes Quadrat (Tiefe − 0.5)², damit auch U-förmige Profile erfasst werden |
| `log_norm_a`, `log_norm_b` | log ‖d_a‖, log ‖d_b‖: Norm der rohen Massenmittel-Differenz (Gl. 2) beider Werte auf diesem Head |
| `score_a`, `score_b` | gespeicherter Ranking-Score der Head-Tabellen beider Werte (`meta.score`, in der Regel AUC); fehlt eine Head-Tabelle, entfällt die Kovariate |
| `r_step0` | r(h) desselben Heads bei step0 (Initialisierungs-Referenz); nicht bei step0 selbst. Im Modell für \|r\| geht \|r(step0)\| ein |

Eine Kovariate entfällt für eine Zeile, wenn sie fehlt, auf dem Head-Satz einen nicht-endlichen Wert hat oder konstant
ist (`pred_covariates` nennt die verwendeten).

*Schätzer:* Ridge-Regression auf standardisierten Kovariaten. Standardisiert wird über den Head-Satz; das nutzt keine
Zielwerte und legt nur die Skala der Strafe fest. Der Achsenabschnitt bleibt unbestraft, die Strafe ist λ = α · n_train
mit α ∈ {100, 10, 1, 0.1, 0.01, 0.001}; α = 100 ist praktisch das reine Mittelwertmodell. α wird je äußerem Fold per
**innerer CV** über die übrigen 4 Folds gewählt (bei Gleichstand die stärkere Strafe). Äußere CV: **5 Folds aus
ganzen Layern** (Layer-Rang mod 5, jeder Fold überspannt die Tiefe, ein Tiefenprofil wird interpoliert statt
extrapoliert). Die Heads eines Layers sind nie gleichzeitig in Training und Test, deshalb kann eine Layer-Häufung nicht
als Vorhersagbarkeit durchsickern. Prüfung (24 × 16 Heads, Layer-Versätze mit ICC 0.8, die nicht von den Kovariaten
abhängen, dazu zwei je Layer konstante Zufallskovariaten, die den Layer kennzeichnen): Mit Folds aus ganzen Layern liegt
das mittlere Held-out-R² bei −0.04, mit Folds aus einzelnen Heads bei +0.12 (der Versatz eines Layers wird an dessen
übrigen Heads gelernt; 300 Nullläufe); mit Head-Folds und einfachen Permutationen wäre der Test in 94 % der
Nullläufe signifikant (`test_layer_blocked_cv_does_not_leak_layer_offsets`). **Held-out-R² = 1 − Σ SS_res / Σ SS_tot**
über die Folds, mit SS_tot um den Trainingsmittelwert des jeweiligen Folds (die Vorhersage ohne Kovariaten). Rauschen
ergibt also im Mittel R² ≤ 0, und negative Werte sind möglich (bei Layer-Häufung deutlich negativ, weil der
Trainingsmittelwert den Versatz eines zurückgehaltenen Layers nicht kennt; die Null hat dieselbe Struktur).
`pred_*_r2_null` ist der Median der Permutationsnull, `pred_*_alpha` der Median des gewählten α, `pred_*_coefs` die
standardisierten Koeffizienten eines Fits auf allen Heads (α mit dem kleinsten CV-Fehler), `pred_*_top` die Kovariate
mit dem größten Betrag. Alle Fits brauchen nur Summen je Fold, deshalb laufen die Permutationen vektorisiert:
1024 Heads × 2000 Permutationen in ≈ 0.5 s. Ein Test vergleicht das Ergebnis mit einer expliziten verschachtelten CV.

*Null und p-Wert:* N_PERM = 2000 Permutationen, die **ganze Layer** auf Layer mit gleicher Head-Zahl verschieben und die
Heads innerhalb jedes Layers mischen (`layer_block_perms`). Für beide Ziele gelten dieselben Ziehungen. Diese
Permutationen bilden eine Gruppe, der Test ist also exakt, sobald die Residuen-Blöcke über gleich große Layer
austauschbar sind. Das gilt für unabhängige Residuen, aber auch für Layer-Versätze, die nicht von den Kovariaten
abhängen: Die Layer-Häufung (d) bleibt in der Null erhalten und wird nicht als Vorhersagbarkeit gezählt.
p = (1 + #{R²_perm ≥ R²})/(1 + N_PERM), einseitig. Einfache Permutationen über alle Heads wären unter Layer-Häufung zu
liberal. Simulation (400 Nullläufe je Zeile, 199 Permutationen, Anteil p ≤ 0.05 für r / |r|):

| Nullfall | Layer-Block-Permutation | einfache Permutation |
|---|---|---|
| unabhängig, 24 × 16 Heads | 0.048 / 0.048 | 0.050 / 0.052 |
| unabhängig, 12 × 12 Heads | 0.058 / 0.040 | 0.062 / 0.045 |
| Layer-Versätze (ICC 0.3), 12 × 12, Normen wachsen mit der Tiefe | 0.055 / 0.040 | **0.107** / 0.043 |
| unabhängig, 20 von 12 × 12 Heads (wie geteilte Top-15 %) | 0.050 / 0.048 | 0.050 / 0.055 |

Breitere Prüfung der Layer-Block-Permutation (Review; Kovariaten wie im echten Lauf strukturiert: Normen und Probe-Scores
wachsen mit der Tiefe, r(step0) unabhängig; 199 Permutationen): 20 Nullzellen mit unabhängigen Residuen, 19–1024 Heads in
12–32 Layern (ganze Modelle 12 × 12, 16 × 8, 24 × 16, 32 × 32 und nach Layern gehäufte Teilmengen wie die geteilten
Top-15 %-Heads), je 400 Läufe, zusammen 7989 Läufe: **0.048 / 0.048**. 11 Zellen mit Layer-Versätzen (ICC 0.3–0.7),
4400 Läufe: **0.051 / 0.052**. Einzelne Zellen streuen zwischen 0.028 und 0.080 (Standardfehler je Zelle 0.011); die
auffälligste Zelle (36 von 12 × 12 Heads, |r| 0.080) ergab mit 2500 neuen Läufen 0.055. Je 1500 Läufe: 24 × 16 mit ICC 0.8
und zwei je Layer konstanten Kovariaten 0.050 / 0.059, 32 × 32 unabhängig 0.051 / 0.049, 12 × 12 mit ICC 0.5
0.042 / 0.055 (Standardfehler 0.006).

*Nicht berechenbar* (NaN) ist (g) bei weniger als 16 Heads, weniger als 5 Layern (jeder Fold muss einen ganzen Layer
zurückhalten) oder ohne verwendbare Kovariate.

*Power und Urteile* (60 Läufe je Fall, B = P = 500, Holm in einer Einzeilen-Familie; „M“ = Anteil
„mean-field-artig“ ohne → mit (g), „V“ = Anteil „vorhersagbar“):

| Fall | Heads, d | R² r / \|r\| | Steigung (KI ∋ −1/2) | M ohne → mit (g) | V |
|---|---|---|---|---|---|
| unabhängiges Rauschen | 24 × 16, 128 | −0.003 / −0.002 | −0.504 (98 %) | 0.97 → 0.95 | 0.03 |
| Rest vom Probe-Score (wahres R² 0.81) | 24 × 16, 128 | +0.80 / −0.004 | −0.503 (97 %) | 0.97 → 0.00 | 1.00 |
| Tiefenprofil cos(π · Tiefe) + Rauschen (wahres R² 0.36) | 24 × 16, 128 | +0.34 / +0.01 | −0.17 (0 %) | 0.00 → 0.00 | 1.00 |
| Betrag ∝ exp(0.6 · log-Norm), Vorzeichen zufällig | 24 × 16, 128 | −0.003 / +0.27 | −0.506 (95 %) | 0.95 → 0.00 | 1.00 |
| aus step0 eingefroren: r = 0.8 · r(step0) + 0.6 · Rauschen | 24 × 16, 128 | +0.62 / +0.34 | −0.497 (98 %) | 0.98 → 0.00 | 1.00 |
| unabhängiges Rauschen | 12 × 12, 64 | −0.009 / −0.006 | −0.506 (95 %) | 0.95 → 0.95 | 0.00 |
| Rest vom Probe-Score | 12 × 12, 64 | +0.79 / −0.008 | −0.500 (100 %) | 1.00 → 0.00 | 1.00 |
| Tiefenprofil | 12 × 12, 64 | +0.29 / −0.000 | −0.18 (0 %) | 0.00 → 0.00 | 0.88 |
| Betrag ∝ Norm | 12 × 12, 64 | −0.011 / +0.23 | −0.509 (97 %) | 0.95 → 0.00 | 1.00 |

Kleine Head-Sätze (20 von 12 × 12 Heads, 200 Läufe, 199 Permutationen, roher p ≤ 0.05): Bei wahrem R² 0.81 liegt die
Power bei 0.92 (mittleres Held-out-R² +0.55), bei wahrem R² 0.5 bei 0.53 (+0.14). Ohne Struktur liegt der Anteil bei 0.05.

*Dünne Head-Sätze mit ungleich besetzten Layern* (typisch für die geteilten Top-15 %-Heads, die sich in wenigen Layern
häufen): Die Layer-Block-Permutation verschiebt einen Layer nur auf einen Layer mit gleicher Head-Zahl. Ein Layer, dessen
Head-Zahl sonst nicht vorkommt, bleibt an seinem Platz, nur seine Heads werden gemischt. Der Test bleibt exakt (die
Permutationen bilden weiter eine Gruppe), verliert aber Power: vor allem für Kovariaten, die innerhalb eines Layers
konstant sind (Tiefe), bei wenigen Heads auch für Head-Eigenschaften. Simulation (Review; Layer-Gewichte der Heads
~ Dirichlet(0.5), also stark gehäuft; 100 Läufe je Zelle, 200 im Nullfall, 199 Permutationen, Anteil roher p ≤ 0.05 für r;
„feste Layer“ = Anteil der Layer mit einer Head-Zahl, die sonst nicht vorkommt):

| Head-Satz | feste Layer | Tiefenprofil, wahres R² 0.3: Layer-Block / einfach | Rest vom Probe-Score, wahres R² 0.3: Layer-Block / einfach | Nullfall ICC 0.5: Layer-Block / einfach |
|---|---|---|---|---|
| 36 von 12 × 12 | 46 % | 0.06 / 0.59 | 0.26 / 0.70 | 0.060 / **0.100** |
| 57 von 24 × 16 | 23 % | 0.41 / 0.84 | 0.70 / 0.88 | 0.065 / 0.090 |
| 153 von 32 × 32 | 27 % | 0.73 / 1.00 | 0.99 / 1.00 | 0.045 / 0.075 |

Bei gleichmäßiger verteilten Heads (Dirichlet(2)) liegt die Power für das Tiefenprofil bei 0.18 / 0.74 / 0.97 (einfach
0.54 / 0.95 / 1.00), die einfache Permutation im Nullfall mit ICC 0.5 bei 0.095–0.110. Die Layer-Block-Permutation bleibt
trotzdem die Wahl: Einfache Permutationen sind unter Layer-Häufung zu liberal, und ein Tiefenprofil ist an die
Layer-Reihenfolge gebunden und fällt deshalb auch (c) (flachere Steigung) und (d) (Layer-Häufung) auf. Bei dünnen
Head-Sätzen sagt „(g) nicht signifikant“ also wenig; ihr Urteil ist ohnehin meist „unterbestimmt“ (Kürzel m).

*Lesehilfe:* Mit `r_step0` als Kovariate gilt auch ein aus der Initialisierung **eingefrorener** Rest als
vorhersagbar: Er ist eine feste Funktion der Initialisierung und kein frisch gezogenes Rauschen, auch wenn er über Heads
unabhängig ist und sich herausmittelt. Welche Kovariate trägt, zeigen `pred_*_top` und `pred_*_coefs`. Die Abhängigkeit
von |r| von den Normen ‖d_a‖, ‖d_b‖ kann teilweise **Schätzrauschen** sein: Bei Heads mit kleiner Norm überwiegt das
Rauschen der Massenmittel-Schätzung, ihr r streut wie N1, während Heads mit großer Norm näher am wahren Winkel liegen.

## p-Werte und Familien

- **Holm je Modell × Checkpoint** über alle Paare, beide Head-Sätze und alle Tests mit p-Wert (Lage N1, Lage step0,
  Streuung N1, Streuung step0, Anteil > 100°, Steigung, ICC, Persistenz vorher, Persistenz zu step143000,
  Vorhersagbarkeit von r und von |r| (g); bis zu 66 Tests). Alles davon ist **deskriptiv**. Die beiden Tests von (g)
  sind seit ihrer Einführung Teil dieser Familie, die Holm-Werte der übrigen Tests sind dadurch etwas größer als vorher.
- **Vorab festgelegter Primärbefund:** (a) und (c) – Lage gegen N1, Lage gegen step0, Mittelungssteigung – auf den
  **geteilten Top-15 %-Heads bei step143000**, je Wertepaar und Modell; Holm innerhalb jedes Modells über diese 3 Paare ×
  3 Tests. Für diese drei Tests verwendet das Urteil der Primärzeilen den Primär-Holm-Wert. (g) gehört **nicht** zur
  Primärfamilie; auch in Primärzeilen zählt für (g) der deskriptive Holm-Wert.
- Permutations-p haben die Auflösung 1/(P + 1) ≈ 0.0005; bei 66 Tests bleibt der kleinste Holm-Wert damit bei
  66/2001 ≈ 0.033 < 0.05. Mit weniger als ≈ 1320 Permutationen (z. B. `--perms 300` im Smoke) kann ein Permutationstest
  nach Holm nicht mehr signifikant werden. Dann heißt auch eine deutlich vorhersagbare Zeile nicht „vorhersagbar“.

## Urteilsregeln (im Code festgelegt, `verdict_of`)

Signifikant heißt: Holm-p < 0.05. Mehrere Etiketten sind möglich; die Spalte `drivers` nennt die Tests, die das Urteil
tragen.

| Etikett | Regel |
|---|---|
| **mean-field-artig** | n ≥ 16 **und** Lage weder gegen N1 noch gegen step0 signifikant **und** KI der Steigung enthält −0.5, ist höchstens 0.5 breit (und die Steigung ist nicht signifikant flacher) **und** der Layer-Test (d) war durchführbar und zeigt keine signifikante Layer-Häufung **und** unimodal **und** die Vorhersagbarkeit (g) war für r und \|r\| berechenbar und ist für keines von beiden signifikant |
| **systematisch** | Lage ≠ N1 oder ≠ step0 (signifikant, Mittel und Median gleiches Vorzeichen) **oder** signifikante Persistenz zum vorherigen Checkpoint |
| **kollektiv** | Steigung signifikant flacher als −0.5 **oder** signifikante Layer-Häufung (ICC) |
| **vorhersagbar** | r **oder** \|r\| ist außerhalb der Stichprobe signifikant vorhersagbar (g); schließt „mean-field-artig“ aus. Passen (a)–(f) zum Mean-Field-Muster, nennt `drivers` das ausdrücklich („… aber der Rest ist vorhersagbar (g)“) |
| **bistabil** | bimodal nach der Regel von (f) |
| **unterbestimmt** | n < 16 (keine Mittelungssteigung) **oder** Lage, Steigung und Form passen zum Mean-Field-Muster (und der Rest ist nicht vorhersagbar), aber der Layer-Test war nicht durchführbar, das Steigungs-KI ist breiter als 0.5 **oder (g) war nicht berechenbar**; Etiketten aus signifikanten Tests bleiben erhalten |
| **uneindeutig** | keine Regel greift (z. B. Steigungs-KI ohne −0.5, aber nach Holm nicht signifikant) |

*(g) als dritte Schranke:* „nicht berechenbar“ zählt ebenso wenig als „nicht vorhersagbar“ wie „nicht testbar“ als
„keine Layer-Häufung“. Fehlt (g) für r oder für |r| (weniger als 16 Heads, weniger als 5 Layer, keine Kovariate), dann
lautet das Urteil bei sonst passendem Mean-Field-Muster „unterbestimmt“ (Kürzel m). In `drivers` steht dann
„Vorhersagbarkeit des Rests (g) nicht berechenbar …“.

*Warum die beiden Schranken:* Ein dünner Head-Satz (meist ein Head je Layer) lässt keinen Layer-Test zu; früher zählte
„nicht testbar“ als „keine Layer-Häufung“. Ein Steigungs-KI von 18–21 Heads ist ≈ 0.7 breit und enthält −0.5 fast immer;
ein KI über 0.5 kann −1/2 (unabhängig) nicht von −1/4 (auf halbem Weg zu vollständig kollektiv, Steigung 0) trennen.
Simulationen (60 Läufe je Zelle, B = P = 500, Layer-Versatz mit wahrer ICC, Gesamtvarianz 2/d, Holm in einer
Einzeilen-Familie):

| Head-Satz | wahre ICC | Layer-Test durchführbar | „mean-field-artig“ alt → neu |
|---|---|---|---|
| 18 von 16 × 8 Heads, d = 256 | 0 / 0.3 / 0.5 | 23 / 18 / 15 % | 93 / 92 / 92 % → 0 % (alle m) |
| 20 von 12 × 12 Heads, d = 64 | 0.5 | 100 % | 70 % → 0 % (KI zu breit) |
| 40 von 24 × 16 Heads, d = 64 | 0.5 | 100 % | 57 % → 53 % |
| alle Heads (12 × 12, 16 × 8, 24 × 16, 32 × 32) | 0 | 100 % | 90–93 % → unverändert |

Für die geteilten Top-15 %-Heads von pythia-160m (k = 21) und pythia-1b (k = 19) heißt das Urteil deshalb meist
„unterbestimmt“ (Kürzel **m**), und die Aussage über alle Heads ist die belastbare. Die dritte Zeile zeigt die
verbleibende Grenze: Ist der Layer-Test durchführbar, aber stehen nur wenige Heads je Layer zur Verfügung, hat er nach
Holm wenig Power. Die Spalte `drivers` nennt den Grund („Layer-Häufung nicht testbar …“, „Steigungs-KI zu breit …“);
„Layer-Häufung nicht testbar“ steht dort auch bei Zeilen mit anderen Etiketten, und die Primärtabelle der
Modell-Zusammenfassung hat eine Spalte „Layer-Test (d)“.

Kürzel in den Verlaufstabellen: M, S, K, **V** (vorhersagbar), B, U, ?; **P** statt S, wenn nur die Persistenz
„systematisch“ auslöst; **m** statt U, wenn Lage, Steigung und Form zum Mean-Field-Muster passen, aber eine der drei
Schranken greift (Layer-Test nicht durchführbar, Steigungs-KI zu breit, (g) nicht berechenbar). P ist der Fall
*eingefrorener Unordnung*: Jeder Head behält sein Residuum vom vorherigen Checkpoint, aber über Heads mittelt es sich
heraus. „mean-field-artig“ und „systematisch (P)“ schließen sich nicht aus, solange der Rest nicht aus r(step0) oder den
anderen Kovariaten vorhersagbar ist. Ist er seit der Initialisierung eingefroren, sagt `r_step0` ihn vorher (g), und die
Zeile heißt „vorhersagbar“ statt „mean-field-artig“.

## Trajektorien und Breiten-Achse

Je Modell und Paar: mittleres r, sd(r)/sd_N1, Mittelungssteigung (je mit KI, beide Head-Sätze) und das Urteil über die 12
Checkpoints (x = Schritt + 1, logarithmisch). Markiert sind step1000 und step2000, das IOI-Tief unter Zufallsniveau nach
Dahiya & Blondin (2026): Boden bei step2000 für 160m/410m, bei step1000 für 1b (dicker gezeichnet). Breiten-Achse:
sd(r)/sd_N1 bei step143000 gegen die nominelle Modellgröße (sobald ≥ 2 Modelle gefunden sind).

## Ausgaben

Je Modell im Wurzelordner des Modells auf Drive:

- `results/meanfield/{modell}_meanfield.csv` – eine Zeile je Checkpoint × Paar × Head-Satz (Spalten u. a. `verdict`,
  `verdict_code`, `drivers`, `status` (primär/deskriptiv), `angle_n/mean_deg/median_deg/std_deg` (Tab.-17-Zahlen),
  `mean_r` + KI, `n1_mean_lo/hi`, `delta_r` + KI, `sd_ratio` + KI, `excess_var`, `frac_gt100` vs. `frac_gt100_n1`, `slope`
  + KI, `slope_raw`, `slope_null_mean`, `n_eff`, `neff_ratio`, `icc`, `rho_prev`, `rho_final`, `kurt`, `gmm_*`,
  Vorhersagbarkeit (g): `pred_covariates`, `pred_n_cov`, `pred_folds`, `pred_r_r2` / `pred_absr_r2` (Held-out-R²),
  `pred_*_r2_null`, `pred_*_alpha`, `pred_*_top`, `pred_*_coefs`, `p_pred_r` / `p_pred_absr`; alle `p_*` mit `p_*_holm`
  bzw. `p_*_primary_holm`, `heads` = die geteilten Heads als flat-Indizes),
- `results/meanfield/{modell}_meanfield.md` – deutsche Zusammenfassung (Primärbefund mit Spalte „Vorhersagbarkeit (g)“,
  Tab.-17-Vergleich, Tabelle „Vorhersagbarkeit des Rests (g)“ beim letzten Checkpoint, Verlauf der Urteile, Treiber,
  Vorbehalte),
- `results/meanfield/{modell}_meanfield.json` – Fingerabdruck der Eingaben (Inhalt aller npz/json, Einstellungen,
  Quelltext des Moduls) für die Idempotenz,
- `figures/meanfield/{modell}_{paar}_trajektorie.png`, `{modell}_step143000_r_hist.png` (Histogramm gegen die exakte
  N1-Dichte), `{modell}_step143000_mittelung.png` (Mittelungsgesetz, log-log).

Bei mehreren Modellen zusätzlich `MyDrive/ValueHeadsPythia/meanfield_summary_{modelle}.csv` und
`meanfield_breite_{modelle}.png`. Alles ist klein (CSV je Modell ~0.1 MB, PNG 100–200 kB) und läuft auf der CPU.

## Ausführen

**Colab (CPU-Laufzeit genügt):** `notebooks/runs/meanfield-angles.ipynb` öffnen, *Laufzeit → Alle ausführen*. Einmal mit
Konto A (pythia-160m, -410m, -1b), einmal mit Konto B (pythia-1.4b, -2.8b). Das Notebook sucht unter
`MyDrive/ValueHeadsPythia` und dessen Unterordnern (zwei Ebenen) nach `results/directions/` (Wurzel von pythia-160m kann
`MyDrive/ValueHeadsPythia` selbst sein). Ein zweiter Lauf ohne neue Daten überspringt alles. Die neuen Dateien müssen im
GitHub-Repo liegen, weil das Notebook das Projekt per `git clone` holt.

**Lokal / Kommandozeile:**

    python -m valueheads_pythia.analysis.meanfield --base /pfad/zu/ValueHeadsPythia        # alle gefundenen Modelle
    python -m valueheads_pythia.analysis.meanfield --base ... --models pythia-1b --force    # ein Modell, neu rechnen
    python -m valueheads_pythia.analysis.meanfield --synthetic /tmp/mf_smoke                # synthetischer Baum + Analyse

**Tests und Smoke:**

    python3 -m pytest tests/test_meanfield.py -q
    python3 scripts/make_meanfield_notebook.py
    VH_NB_SMOKE=1 VH_NB_PROJECT=$PWD VH_NB_BASE=/tmp/mf_nb_smoke \
      jupyter nbconvert --to notebook --execute --output /tmp/mf_nb_smoke_executed.ipynb notebooks/runs/meanfield-angles.ipynb

Der Smoke legt einen synthetischen Baum mit bekannter Struktur an (pythia-160m-Geometrie im Basisordner, pythia-1b in
einem Unterordner; harmless und helpful teilen eine mit dem Training wachsende Komponente, honesty ist unabhängig) und
liest keine echten Daten. Die synthetischen Richtungen sind die step0-Richtungen plus eine kleine Drift. r(h) ist dort
also auch bei step143000 stark aus `r_step0` vorhersagbar (Held-out-R² ≈ 0.5). Mit 2000 Permutationen heißen diese
Zeilen deshalb „vorhersagbar“. Mit wenigen Permutationen (`--perms 300`) erreicht (g) nach Holm keine Signifikanz (siehe
„p-Werte und Familien“).

## Vorbehalte

- **„mean-field-artig“ ist eine statistische Signatur**: Die Residuen verhalten sich wie unabhängiges Rauschen, das sich
  über Heads herausmittelt. Es ist **keine** Aussage über das Regime der Trainings-Parametrisierung im Sinne der
  Mean-Field-Theorie gegenüber dem Lazy/NTK-Regime.
- Rauschen der Massenmittel-Schätzung zieht alle Winkel Richtung 90° und dekorreliert die Residuen: Überschuss-Streuung,
  Lage ≠ 0 und Layer-Häufung sind konservativ; eine „mean-field-artige“ Signatur kann dagegen teilweise Schätzrauschen
  sein. Ohne Features ist dieser Anteil nicht mehr bestimmbar.
- Persistenz ist früh im Training fast trivial (das Netz ändert sich zwischen step0 und step512 kaum), und alle
  Checkpoints verwenden dieselben Texte; ihr Schätzrauschen ist über Checkpoints korreliert und trägt zur Persistenz bei.
- Die geteilten Top-15 %-Mengen sind je Checkpoint neu bestimmt und oft klein (pythia-160m: k = 21 je Wert, pythia-1b:
  k = 19); bei n < 16 gibt es keine Mittelungssteigung, und auch darüber fehlt meist der Layer-Test, und das
  Steigungs-KI ist breiter als 0.5 („unterbestimmt“, Kürzel m). Das gilt auch für den vorab festgelegten Primärbefund
  dieser Modelle: Lage (a) und Steigung (c) werden dort berichtet und getestet, ein Urteil „mean-field-artig“ geben sie
  allein nicht her. Die Aussage über alle Heads ist dort die belastbarere.
- Das Mittelungsgesetz (c) ist blind für einen Anteil, den alle Heads teilen (siehe (c)); ob ein solcher Anteil vorliegt,
  entscheidet allein der Lagetest (a). Eine kleine gemeinsame Verschiebung unterhalb der Nachweisgrenze von (a) kann
  deshalb als „mean-field-artig“ durchgehen.
- (c) ist auch blind für einen Rest, der von Head-Eigenschaften abhängt, die nicht an die Layer-Reihenfolge gebunden sind.
  Einen solchen Rest erkennt nur (g), und auch nur, wenn er von den verwendeten Kovariaten abhängt, und zwar linear oder
  quadratisch in der Tiefe. Ein „mean-field-artig“ heißt also: kein mit diesen Kovariaten vorhersagbarer Rest, nicht
  „kein Rest mit Struktur“. Bei den kleinen geteilten Top-15 %-Mengen hat (g) wenig Power (20 Heads, wahres R² 0.5:
  Power ≈ 0.5 vor Holm), und noch weniger, wenn sich die Heads in wenigen Layern häufen (siehe (g), dünne
  Head-Sätze mit ungleich besetzten Layern).
- (g) ist deskriptiv (keine Primärfamilie). Mit `r_step0` zählt ein aus der Initialisierung eingefrorener Rest als
  vorhersagbar. Eine Abhängigkeit des Betrags |r| von den Normen der Massenmittel-Differenzen kann teilweise vom
  Schätzrauschen kommen (siehe (g)).
- step0 als N2 kennt nur das untrainierte Netz *dieses* Modells; Qwen3-4B (Tab. 17) hat keinen step0-Checkpoint, der
  Vergleich mit dem Paper bleibt dort auf N1 beschränkt.
- Die Bimodalitätsregel ist eine feste Schwelle, kein Test mit p-Wert; „bistabil“ ist deskriptiv.
- Multiples Testen: nur der Primärbefund ist konfirmatorisch, alle Verläufe sind deskriptiv zu lesen.
