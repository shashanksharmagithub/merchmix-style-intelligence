# AI Concept Generation Trace

Final artifact:

```text
outputs/generated_concepts.png
```

Conservative first pass (retained for comparison):

```text
outputs/generated_concepts_conservative.png
```

Fallback deterministic board:

```text
outputs/generated_concepts_fallback.png
```

## Source References

1. Rank 1 style `610776`, source image `data/raw/images/061/0610776006.jpg`
2. Rank 2 style `717490`, source image `data/raw/images/071/0717490002.jpg`
3. Rank 3 style `599580`, source image `data/raw/images/059/0599580001.jpg`

## Prediction-To-Concept Link

| Rank | Predicted style | Model-backed success signals | Final concept | Meaningful changes |
| ---: | --- | --- | --- | --- |
| 1 | `610776` Tilly (1) | 6,715 recent units; 22,407 historical units; demand is 419.69x its garment-group median | Botanical Gather Camp Tee | Cropped boxy shape, camp collar, engineered floral placement, asymmetric hem, gathered side |
| 2 | `717490` Cat Tee. | 6,419 recent units; 18,226 historical units; demand is 401.19x its garment-group median | Arc-Panel Jersey Tee | Mock neck, curved panel seams, tonal eyelet inserts, structured cuffs, high-low hem |
| 3 | `599580` Timeless Midrise Brief | 8,049 recent units; 31,118 historical units; demand is 365.86x its garment-group median | Contour-Wrap Swimwear Brief | Asymmetric high waist, diagonal wrap waistband, mixed matte/rib panels, tonal stitching, side adjuster |

The exact grouped counterfactual explanation deltas are recorded in `outputs/style_explanations.md` and `outputs/concept_briefs.json`. They are diagnostic, non-additive, and not causal.

## Final Adventurous Prompt

Use case: product-mockup

Asset type: adventurous next-season fashion concept board for a data-science merchandising submission

Primary request: Create one wide 3-panel premium catalogue board showing exactly three newly designed garments, in this exact left-to-right rank order. Use the three supplied catalogue images only as source-style references, not as products to copy.

Panel 1, Rank 1, inspired by reference image 1: Botanical Gather Camp Tee. Keep the off-white ground and blue/orange/teal/green botanical language, but transform it into a clearly new boxy cropped camp-collar top with a bolder engineered floral placement, asymmetric curved hem, and dramatic gathered side panel.

Panel 2, Rank 2, inspired by reference image 2: Arc-Panel Jersey Tee. Keep the clean white jersey foundation, but create an architectural premium top with compact mock neck, strongly curved front panel seams, tonal eyelet inserts, structured cuffs, and a controlled high-low hem.

Panel 3, Rank 3, inspired by reference image 3: Contour-Wrap Swimwear Brief. Keep the black swimwear-bottom context, but create an asymmetric high-waist wrap silhouette with sculpted diagonal waistband, clearly mixed matte and ribbed panels, tonal topstitching, and one refined dark metal side adjuster.

Composition: three equal vertical panels separated by thin neutral dividers; one full garment centered in each panel; front-facing flat-lay/e-commerce presentation; all products completely visible with generous breathing room; exact left-to-right order floral top, white architectural tee, black wrap brief.

Scene/backdrop: clean cool-white studio background, soft realistic shadows, crisp fabric texture.

Style: sophisticated adventurous commercial fashion design, photorealistic product development render, sharper and more directional than basic catalogue garments.

Text: no text, no logos, no watermarks, no people, no hangers, no extra accessories.

Critical constraints: exactly three garments, no duplicates, no model bodies, do not reproduce the source garments unchanged, preserve the specified category for each panel, do not swap panel order.

## Earlier Conservative Prompt

Use case: product-mockup

Asset type: final assignment concept image, one board containing three next-season apparel concepts in model rank order

Primary request: Create one polished fashion concept board showing exactly three new product concepts, each inspired by one reference image. Arrange left to right in this exact order: rank 1 black swimwear bottom concept, rank 2 off-white floral t-shirt concept, rank 3 white t-shirt concept. The concepts should keep the strongest commercial visual idea of the source style but introduce meaningful design changes, not copies.

Input images: Image 1 is reference for a successful black mid-rise swimwear bottom; Image 2 is reference for a successful off-white floral jersey t-shirt; Image 3 is reference for a successful white jersey basic t-shirt.

Scene/backdrop: clean neutral studio background, editorial catalogue board, soft natural shadows, no visible model or mannequin.

Subject: Three separate flat-lay or ghost-mannequin product concepts arranged left to right: 1) next-season black swimwear bottom with refined waistband, subtle ribbed texture, and modernized coverage/proportion; 2) next-season off-white t-shirt that retains floral all-over appeal but updates the print scale, placement, neckline, and hem proportion; 3) next-season white t-shirt with updated neckline, subtle seam/finish detail, and premium jersey texture.

Style/medium: realistic fashion product photography, catalogue-quality, commercially plausible mass-market apparel.

Composition/framing: wide horizontal board with three equally sized product panels, full products visible, generous padding, consistent scale and lighting across all three concepts.

Lighting/mood: clean bright studio lighting, premium but practical retail presentation.

Color palette: neutral background, black swimwear, off-white floral top with restrained blue/orange/green floral accents, clean white top.

Constraints: Make three new concepts inspired by the references; do not reproduce the exact source products. No logos, no readable brand tags, no text labels, no watermarks, no extra products, no people, no hangers.
