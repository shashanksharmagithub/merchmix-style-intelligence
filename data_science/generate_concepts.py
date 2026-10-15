from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from data_science.config import CONFIG, ensure_project_dirs


FINAL_CONCEPT_DIRECTIONS = {
    "599580": {
        "concept_name": "Contour-Wrap Swimwear Brief",
        "retained_elements": [
            "black core colour",
            "proven swimwear-bottom category",
            "commercially wearable coverage",
        ],
        "meaningful_changes": [
            "asymmetric high-waist wrap silhouette",
            "sculpted diagonal waistband",
            "mixed matte and ribbed panels",
            "tonal topstitching and a refined side adjuster",
        ],
        "design_direction": (
            "Evolve the proven black mid-rise brief into an asymmetric high-waist wrap shape with "
            "a sculpted diagonal waistband, mixed matte/ribbed panels, tonal topstitching, and one "
            "refined side adjuster."
        ),
    },
    "610776": {
        "concept_name": "Botanical Gather Camp Tee",
        "retained_elements": [
            "off-white ground",
            "blue, orange, teal, and green botanical palette",
            "easy jersey-top appeal",
        ],
        "meaningful_changes": [
            "boxy cropped proportion",
            "short camp collar",
            "asymmetric curved hem",
            "gathered side panel and engineered floral placement",
        ],
        "design_direction": (
            "Turn the successful floral basic into a boxy cropped camp-collar top with a bolder "
            "engineered botanical placement, asymmetric curved hem, and gathered side panel."
        ),
    },
    "717490": {
        "concept_name": "Arc-Panel Jersey Tee",
        "retained_elements": [
            "clean white colour cue",
            "soft jersey foundation",
            "versatile everyday-top function",
        ],
        "meaningful_changes": [
            "compact mock neckline",
            "architectural curved front seams",
            "tonal eyelet inserts",
            "structured cuffs and controlled high-low hem",
        ],
        "design_direction": (
            "Rebuild the plain white tee as an architectural premium jersey top with a compact mock "
            "neck, curved panel seams, tonal eyelet inserts, structured cuffs, and a controlled "
            "high-low hem."
        ),
    },
}


def _concept_for_style(style: dict) -> dict:
    category = style.get("category") or "garment"
    colour = style.get("colour") or "core colour"
    group = style.get("garment_group") or style.get("attributes", {}).get("product_group", "fashion")
    reasons = style.get("model_explanation", [])
    final_direction = FINAL_CONCEPT_DIRECTIONS.get(str(style["style_id"]))
    if final_direction:
        concept_name = final_direction["concept_name"]
        design_direction = final_direction["design_direction"]
        retained_elements = final_direction["retained_elements"]
        meaningful_changes = final_direction["meaningful_changes"]
    else:
        concept_name = f"Next-season {colour.lower()} {category.lower()}"
        design_direction = (
            f"Retain the winning {category.lower()} silhouette and {colour.lower()} colour cue, "
            f"then modernize it with cleaner finishings, fresher proportion, and season-ready detailing."
        )
        retained_elements = [f"{colour} colour family", f"{category} product context"]
        meaningful_changes = ["updated finishings", "new proportion", "season-ready detailing"]

    return {
        "style_id": style["style_id"],
        "rank": style.get("rank"),
        "source_image_url": style.get("image_url"),
        "source_style": style.get("name", ""),
        "concept_name": concept_name,
        "retained_elements": retained_elements,
        "meaningful_changes": meaningful_changes,
        "design_direction": design_direction,
        "success_link": reasons,
        "linkage_summary": (
            f"The {concept_name} carries forward the source style's {colour.lower()} colour and "
            f"{group.lower()} commercial context, while changing silhouette and construction so the "
            "result reads as a new next-season product rather than a copy."
        ),
        "prompt": (
            f"Create the {concept_name}, inspired by a successful H&M {category} style. "
            f"Retain: {', '.join(retained_elements)}. Introduce: {', '.join(meaningful_changes)}. "
            "Do not copy the source product. "
            "Clean catalogue lighting, full product visible, neutral background."
        ),
    }


def build_concept_briefs() -> list[dict]:
    ensure_project_dirs()
    if not CONFIG.top_styles_path.exists():
        raise FileNotFoundError("Missing top styles. Run data_science.train first.")
    payload = json.loads(CONFIG.top_styles_path.read_text(encoding="utf-8"))
    concepts = [_concept_for_style(style) for style in payload["styles"]]
    CONFIG.concept_brief_path.write_text(json.dumps({"concepts": concepts}, indent=2), encoding="utf-8")
    return concepts


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if draw.textbbox((0, 0), candidate, font=font)[2] <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_product_concept(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    concept: dict,
    accent: str,
) -> None:
    x1, y1, x2, y2 = box
    draw.rectangle(box, fill=accent)
    category = concept["concept_name"].lower()
    cx = (x1 + x2) // 2
    top = y1 + 54
    ink = "#1f2a2e"
    fabric = "#f9f7f2"

    if "swimwear" in category:
        draw.polygon(
            [
                (cx - 116, top + 70),
                (cx - 42, top + 42),
                (cx, top + 92),
                (cx + 42, top + 42),
                (cx + 116, top + 70),
                (cx + 72, top + 212),
                (cx + 20, top + 188),
                (cx, top + 222),
                (cx - 20, top + 188),
                (cx - 72, top + 212),
            ],
            fill=fabric,
            outline=ink,
        )
        draw.arc((cx - 120, top + 8, cx + 120, top + 140), 18, 162, fill=ink, width=4)
        draw.line((cx - 76, top + 118, cx + 76, top + 118), fill=ink, width=3)
    else:
        draw.polygon(
            [
                (cx - 148, top + 42),
                (cx - 80, top),
                (cx - 36, top + 42),
                (cx + 36, top + 42),
                (cx + 80, top),
                (cx + 148, top + 42),
                (cx + 108, top + 116),
                (cx + 64, top + 96),
                (cx + 60, top + 238),
                (cx - 60, top + 238),
                (cx - 64, top + 96),
                (cx - 108, top + 116),
            ],
            fill=fabric,
            outline=ink,
        )
        draw.arc((cx - 42, top + 8, cx + 42, top + 72), 10, 170, fill=ink, width=4)
        draw.line((cx - 50, top + 154, cx + 50, top + 154), fill=ink, width=3)
        draw.line((cx - 40, top + 188, cx + 40, top + 188), fill=ink, width=3)


def make_concept_board() -> Path:
    """Create a clean board from generated concept briefs.

    This is a deterministic fallback board. For the final submission, replace or augment it with
    an AI-generated concept image once the top style images are available.
    """
    concepts = build_concept_briefs()
    width, height = 1800, 1080
    board = Image.new("RGB", (width, height), "#f7f4ef")
    draw = ImageDraw.Draw(board)
    title_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 40)
    heading_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 26)
    body_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 21)
    small_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 17)
    label_font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 16)

    draw.text((50, 36), "Next-Season Product Concepts", fill="#1f2a2e", font=title_font)
    draw.text((50, 88), "Predicted style - key success factors - new product concept", fill="#526066", font=body_font)

    palette = ["#d56f3e", "#40798c", "#2f6f4e"]
    card_w = 540
    for i, concept in enumerate(concepts):
        x = 50 + i * (card_w + 40)
        y = 150
        draw.rounded_rectangle((x, y, x + card_w, y + 850), radius=8, fill="#ffffff", outline="#d8d2c9", width=2)
        _draw_product_concept(draw, (x + 28, y + 34, x + card_w - 28, y + 354), concept, palette[i % len(palette)])

        yy = y + 386
        draw.text((x + 28, yy), concept["concept_name"], fill="#1f2a2e", font=heading_font)
        yy += 44
        draw.text((x + 28, yy), f"Source style_id: {concept['style_id']}", fill="#667277", font=small_font)
        yy += 34

        draw.text((x + 28, yy), "Design direction", fill="#1f2a2e", font=label_font)
        yy += 26
        lines = _wrap_text(draw, concept["design_direction"], body_font, card_w - 56)
        for line in lines[:6]:
            draw.text((x + 28, yy), line, fill="#39464d", font=body_font)
            yy += 30
        yy += 20

        draw.text((x + 28, yy), "Why it connects to the winner", fill="#1f2a2e", font=label_font)
        yy += 28
        for reason in concept.get("success_link", [])[:3]:
            reason_lines = _wrap_text(draw, f"- {reason}", small_font, card_w - 56)
            for line in reason_lines[:2]:
                draw.text((x + 28, yy), line, fill="#667277", font=small_font)
                yy += 24

    board.save(CONFIG.fallback_concept_image_path)
    if not CONFIG.concept_image_path.exists():
        board.save(CONFIG.concept_image_path)
    return CONFIG.fallback_concept_image_path


if __name__ == "__main__":
    print(f"Wrote briefs to {CONFIG.concept_brief_path}")
    print(f"Wrote board to {make_concept_board()}")
