import { ImageResponse } from "next/og";

export const runtime = "edge";
export const alt = "Villa Mas Nou — Platja d'Aro, Costa Brava";
export const size = { width: 1200, height: 630 };
export const contentType = "image/png";

export default async function Image() {
  return new ImageResponse(
    (
      <div
        style={{
          width: "100%",
          height: "100%",
          display: "flex",
          flexDirection: "column",
          justifyContent: "space-between",
          padding: "70px",
          background:
            "linear-gradient(180deg, #faf6ec 0%, #efe1c0 60%, #d9b58b 100%)",
          color: "#3d2f24",
          fontFamily: "Georgia, serif",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: "16px",
            fontSize: "20px",
            letterSpacing: "0.25em",
            textTransform: "uppercase",
            color: "#7a5f43",
          }}
        >
          <span style={{ color: "#c97b5b", fontSize: "26px" }}>✦</span>
          <span>Platja d&apos;Aro · Costa Brava</span>
        </div>

        <div style={{ display: "flex", flexDirection: "column" }}>
          <div
            style={{
              fontSize: "120px",
              lineHeight: 0.95,
              letterSpacing: "-0.02em",
              fontStyle: "italic",
              color: "#3d2f24",
              display: "flex",
            }}
          >
            Villa Mas Nou
          </div>
          <div
            style={{
              marginTop: "28px",
              fontSize: "32px",
              maxWidth: "780px",
              color: "#5c4732",
              display: "flex",
            }}
          >
            White walls, pine, and the bay below.
          </div>
        </div>

        <div
          style={{
            display: "flex",
            justifyContent: "space-between",
            alignItems: "center",
            fontSize: "20px",
            color: "#7a5f43",
          }}
        >
          <span>Sleeps 8 · Private pool · 5 min to the cove</span>
          <span style={{ display: "flex", alignItems: "center", gap: "12px" }}>
            <span
              style={{
                display: "inline-block",
                width: "40px",
                height: "1px",
                background: "#c97b5b",
              }}
            />
            <span style={{ color: "#c97b5b", fontStyle: "italic" }}>
              villamasnou
            </span>
          </span>
        </div>
      </div>
    ),
    { ...size },
  );
}
