import React, { useState } from "react";
import "./index.css";

const API_URL = "http://127.0.0.1:8000/analyze";

const CHALLENGE_PHRASES = [
  "Hold position at checkpoint",
  "Authentication required for override",
  "Alpha team proceed to sector three",
  "Secure channel only, confirm identity",
  "Abort mission on my signal",
  "Confirm identity and continue",
  "Standby for encrypted transmission",
  "Command authority verification required",
  "Do not execute until code is verified",
  "Report status from sector seven immediately",
];

function App() {
  const [file, setFile] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [loading, setLoading] = useState(false);

  const [juryMode, setJuryMode] = useState(true);
  const [similarityThr, setSimilarityThr] = useState(0.8);
  const [anomalyThr, setAnomalyThr] = useState(0.0);
  const [challengeEnabled, setChallengeEnabled] = useState(true);
  const [challengePhrase, setChallengePhrase] = useState("");

  const [result, setResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [tab, setTab] = useState("simple"); // "simple" | "expert"

  const onFileChange = (e) => {
    const f = e.target.files[0];
    setFile(f);
    if (f) {
      const url = URL.createObjectURL(f);
      setAudioUrl(url);
    } else {
      setAudioUrl(null);
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      alert("Select an audio file first.");
      return;
    }
    setLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });
      if (!res.ok) {
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();

      // Backend verdict
      const probReal = Number(data.prob_real ?? 0);
      const probFake = Number(data.prob_fake ?? 0);
      const backendVerdict = (data.verdict || "").toLowerCase();
      const isoScore = data.iso_score;
      const bestMatch = data.best_match;
      const similarity = data.similarity;

      // Frontend decision (uses sliders)
      let verdict = backendVerdict;
      let reasons = [];

      if (probReal < 0.6) {
        reasons.push("Spoof model: low real probability");
      }
      if (isoScore !== null && isoScore < anomalyThr) {
        reasons.push("Anomaly detector: unusual embedding");
      }
      if (data.has_enrollment && similarity !== null && similarity < similarityThr) {
        reasons.push("Speaker mismatch: low similarity");
      }

      let baseVerdict = "ACCEPT";
      if (backendVerdict === "fake" && probFake > 0.6) {
        baseVerdict = "REJECT";
      }

      let finalVerdict;
      if (baseVerdict === "ACCEPT" && reasons.length === 0) {
        finalVerdict = "ACCEPT";
      } else if (baseVerdict === "REJECT") {
        finalVerdict = "REJECT";
      } else {
        finalVerdict = "SUSPICIOUS";
      }

      // Risk level (same idea as Streamlit)
      let riskLevel = 0;
      if (probReal < 0.6) riskLevel += 1;
      if (isoScore !== null && isoScore < anomalyThr) riskLevel += 1;
      if (data.has_enrollment && similarity !== null && similarity < similarityThr)
        riskLevel += 1;

      const riskString = riskLevel === 0 ? "Low" : riskLevel === 1 ? "Medium" : "High";

      const merged = {
        ...data,
        prob_real: probReal,
        prob_fake: probFake,
        verdict_frontend: finalVerdict,
        risk_level_frontend: riskString,
        reasons,
        file_name: file.name,
        similarity_threshold_used: similarityThr,
        anomaly_threshold_used: anomalyThr,
      };

      setResult(merged);
      setHistory((prev) => [
        {
          file: file.name,
          verdict: finalVerdict,
          probReal,
          probFake,
          risk: riskString,
        },
        ...prev,
      ]);
    } catch (err) {
      console.error(err);
      alert("Error calling backend: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleGeneratePhrase = () => {
    const idx = Math.floor(Math.random() * CHALLENGE_PHRASES.length);
    setChallengePhrase(CHALLENGE_PHRASES[idx]);
  };

  const verdictStyle = (() => {
    if (!result) return { cls: "", label: "" };
    const v = result.verdict_frontend;
    if (v === "ACCEPT") return { cls: "pill-real", label: "✅ AUTHENTIC" };
    if (v === "REJECT") return { cls: "pill-fake", label: "❌ SUSPECTED SPOOF" };
    return { cls: "pill-susp", label: "⚠️ SUSPICIOUS / REVIEW" };
  })();

  return (
    <div className="shell">
      {/* LEFT PANEL – controls & input */}
      <aside className="sidebar">
        <h2 className="sidebar-title">Controls</h2>

        <label className="sidebar-check">
          <input
            type="checkbox"
            checked={juryMode}
            onChange={(e) => setJuryMode(e.target.checked)}
          />
          Jury demo mode (simplified)
        </label>

        <div className="sidebar-section">
          <div className="sidebar-section-title">Decision thresholds</div>

          <div className="slider-block">
            <div className="slider-label">
              Speaker similarity (cosine): {similarityThr.toFixed(2)}
            </div>
            <input
              type="range"
              min="0.6"
              max="0.99"
              step="0.01"
              value={similarityThr}
              onChange={(e) => setSimilarityThr(parseFloat(e.target.value))}
            />
            <div className="slider-hint">
              Higher = stricter identity match (fewer impostors, more false rejects).
            </div>
          </div>

          <div className="slider-block">
            <div className="slider-label">
              Anomaly score threshold: {anomalyThr.toFixed(2)}
            </div>
            <input
              type="range"
              min="-1.5"
              max="1.5"
              step="0.05"
              value={anomalyThr}
              onChange={(e) => setAnomalyThr(parseFloat(e.target.value))}
            />
            <div className="slider-hint">
              Lower = detector flags more samples as anomalous.
            </div>
          </div>
        </div>

        <div className="sidebar-section">
          <label className="sidebar-check">
            <input
              type="checkbox"
              checked={challengeEnabled}
              onChange={(e) => setChallengeEnabled(e.target.checked)}
            />
            Enable challenge–response concept
          </label>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">Demo examples</div>
          <div className="sidebar-hint">
            (For now, load your own files; example buttons can be wired later.)
          </div>
        </div>

        <div className="sidebar-section">
          <div className="sidebar-section-title">Model status</div>
          <div className="sidebar-hint">
            Enrolled identities: {result?.has_enrollment ? "loaded" : "0 (demo-only)"}
          </div>
        </div>
      </aside>

      {/* RIGHT PANEL – main app */}
      <main className="main">
        <header className="header">
          <div>
            <h1>AI-Based Voice Authentication & Anti-Deepfake Prototype</h1>
            <p className="header-sub">
              Scenario: authenticating critical voice commands in a secure communication
              system. Multi-layer defense against deepfake / spoofed audio.
            </p>
            <ol className="header-list">
              <li>Spoof detector – ML model classifies human vs synthetic/spoofed speech</li>
              <li>Anomaly detector – checks if the signal embedding is unusual</li>
              <li>Speaker matching – cosine similarity vs enrolled voiceprints</li>
              <li>
                Decision logic – outputs <b>ACCEPT / REJECT / SUSPICIOUS</b>
              </li>
            </ol>
          </div>
        </header>

        <div className="tabs">
          <button
            className={`tab-btn ${tab === "simple" ? "tab-active" : ""}`}
            onClick={() => setTab("simple")}
          >
            ⚡ Quick Verdict
          </button>
          <button
            className={`tab-btn ${tab === "expert" ? "tab-active" : ""}`}
            onClick={() => setTab("expert")}
          >
            🧠 Expert Analysis
          </button>
        </div>

        {tab === "simple" && (
          <section className="panel-grid">
            {/* LEFT: Input + challenge */}
            <div className="panel">
              <h2 className="panel-title">1️⃣ Provide audio: upload</h2>
              <p className="panel-sub">
                Upload an audio file (WAV / MP3 / FLAC / OGG / M4A). Use a real recording
                or a TTS/deepfake clip.
              </p>

              <div className="input-box">
                <input
                  type="file"
                  accept="audio/*"
                  onChange={onFileChange}
                  className="file-input"
                />
                <button className="button" onClick={handleAnalyze} disabled={loading}>
                  {loading ? "Analyzing…" : "▶ Analyze"}
                </button>
              </div>

              {audioUrl && (
                <div className="audio-box">
                  <audio controls src={audioUrl} style={{ width: "100%" }} />
                </div>
              )}

              {challengeEnabled && (
                <div className="challenge-box">
                  <div className="challenge-top">
                    <div>
                      <div className="challenge-label">Challenge–Response (concept)</div>
                      <p className="challenge-hint">
                        Click generate to simulate a secure random phrase the caller must
                        repeat.
                      </p>
                    </div>
                    <button className="button secondary" onClick={handleGeneratePhrase}>
                      Generate phrase
                    </button>
                  </div>
                  {challengePhrase && (
                    <pre className="challenge-phrase">{challengePhrase}</pre>
                  )}
                </div>
              )}
            </div>

            {/* RIGHT: Verdict & risk */}
            <div className="panel">
              <h2 className="panel-title">3️⃣ Verdict & Risk Overview</h2>

              {!result && (
                <div className="info-box">
                  Upload and analyze a sample to see the verdict here.
                </div>
              )}

              {result && (
                <>
                  <div
                    className={`verdict-card ${
                      result.verdict_frontend === "REJECT"
                        ? "fake"
                        : result.verdict_frontend === "SUSPICIOUS"
                        ? "susp"
                        : "real"
                    }`}
                  >
                    <span className={`verdict-pill ${verdictStyle.cls}`}>
                      {verdictStyle.label}
                    </span>
                    <div className="verdict-main-row">
                      <div className="verdict-main">
                        <div className="verdict-title">
                          {result.verdict_frontend === "ACCEPT"
                            ? "Access granted – voice accepted"
                            : result.verdict_frontend === "REJECT"
                            ? "Access denied – spoof suspected"
                            : "Proceed with caution – suspicious sample"}
                        </div>
                        <div className="verdict-sub">
                          File: <code>{result.file_name}</code>
                        </div>
                      </div>
                      <div className="verdict-metrics">
                        <div>
                          <div className="metric-label">Prob. REAL</div>
                          <div className="metric-value">
                            {result.prob_real.toFixed(2)}
                          </div>
                        </div>
                        <div>
                          <div className="metric-label">Prob. FAKE</div>
                          <div className="metric-value">
                            {result.prob_fake.toFixed(2)}
                          </div>
                        </div>
                        <div>
                          <div className="metric-label">Risk</div>
                          <div className="metric-value">
                            {result.risk_level_frontend}
                          </div>
                        </div>
                      </div>
                    </div>
                    {!juryMode && result.reasons?.length > 0 && (
                      <ul className="reason-list">
                        {result.reasons.map((r, i) => (
                          <li key={i}>{r}</li>
                        ))}
                      </ul>
                    )}
                  </div>

                  <div className="history-section">
                    <h3 className="history-title">Session History</h3>
                    {history.length === 0 ? (
                      <div className="history-empty">
                        No samples analyzed yet in this session.
                      </div>
                    ) : (
                      <div className="history-table-wrap">
                        <table className="history-table">
                          <thead>
                            <tr>
                              <th>#</th>
                              <th>File</th>
                              <th>Verdict</th>
                              <th>Prob R</th>
                              <th>Prob F</th>
                              <th>Risk</th>
                            </tr>
                          </thead>
                          <tbody>
                            {history.map((h, i) => (
                              <tr key={i}>
                                <td>{history.length - i}</td>
                                <td>{h.file}</td>
                                <td>
                                  <span
                                    className={`tag ${
                                      h.verdict === "ACCEPT"
                                        ? "tag-real"
                                        : h.verdict === "REJECT"
                                        ? "tag-fake"
                                        : "tag-susp"
                                    }`}
                                  >
                                    {h.verdict}
                                  </span>
                                </td>
                                <td>{h.probReal.toFixed(2)}</td>
                                <td>{h.probFake.toFixed(2)}</td>
                                <td>{h.risk}</td>
                              </tr>
                            ))}
                          </tbody>
                        </table>
                      </div>
                    )}
                  </div>
                </>
              )}
            </div>
          </section>
        )}

        {tab === "expert" && (
          <section className="panel-grid expert">
            {!result ? (
              <div className="panel">
                <h2 className="panel-title">Expert Analysis</h2>
                <div className="info-box">
                  Analyze a sample in <b>Quick Verdict</b> first.
                </div>
              </div>
            ) : (
              <>
                <div className="panel">
                  <h2 className="panel-title">Waveform & Spectrogram</h2>
                  {result.plot_image ? (
                    <img
                      src={result.plot_image}
                      alt="Waveform and spectrogram"
                      className="plot-image"
                    />
                  ) : (
                    <div className="info-box">
                      No plot image returned from backend.
                    </div>
                  )}
                </div>

                <div className="panel">
                  <h2 className="panel-title">Internal Metrics (last sample)</h2>
                  <ul className="metrics-list">
                    <li>Prob. real: {result.prob_real.toFixed(3)}</li>
                    <li>Prob. fake: {result.prob_fake.toFixed(3)}</li>
                    <li>
                      Anomaly score:{" "}
                      {result.iso_score !== null ? result.iso_score.toFixed(3) : "None"}
                    </li>
                    <li>Risk level (frontend): {result.risk_level_frontend}</li>
                    <li>
                      Best match:{" "}
                      {result.best_match !== null ? result.best_match : "None"}
                    </li>
                    <li>
                      Similarity:{" "}
                      {result.similarity !== null
                        ? result.similarity.toFixed(3)
                        : "None"}
                    </li>
                  </ul>

                  {!juryMode && (
                    <div className="raw-json">
                      Raw JSON response:
                      <pre>{JSON.stringify(result, null, 2)}</pre>
                    </div>
                  )}
                </div>
              </>
            )}
          </section>
        )}
      </main>
    </div>
  );
}

export default App;
src/index.css