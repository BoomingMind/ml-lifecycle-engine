from __future__ import annotations

"""
Generate an animated GIF that simulates how DefectGuard works end to end.

Why keep this as code instead of manually recording a GIF?

- The asset becomes reproducible.
- Future updates can regenerate the demo with one command.
- The visuals stay consistent with the product story in the README.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont


WIDTH = 1200
HEIGHT = 720
OUT_PATH = Path("docs/assets/defectguard-workflow.gif")

BG_TOP = (11, 18, 32)
BG_BOTTOM = (15, 31, 58)
PANEL = (17, 26, 46)
PANEL_ALT = (22, 35, 61)
BORDER = (44, 70, 115)
TEXT = (235, 241, 255)
MUTED = (167, 182, 214)
ACCENT = (94, 234, 212)
ACCENT_DARK = (13, 84, 76)
WARN = (251, 191, 36)
SUCCESS = (74, 222, 128)
DANGER = (251, 113, 133)
WHITE = (255, 255, 255)


@dataclass(frozen=True)
class Scene:
    step: int
    title: str
    subtitle: str
    real_world: list[str]
    system_actions: list[str]
    outputs: list[str]
    footer: str
    status_label: str
    status_color: tuple[int, int, int]


SCENES = [
    Scene(
        step=0,
        title="Factory cameras capture product images",
        subtitle="A production line continuously produces images of components that need inspection.",
        real_world=[
            "Camera watches metal, PCB, or textile surfaces",
            "Operators want faster and more consistent inspection",
            "Raw images arrive before any model can help",
        ],
        system_actions=[
            "Data lands in raw storage",
            "Dataset config and manifests describe where files live",
            "The pipeline gets ready for validation and training",
        ],
        outputs=[
            "Raw image batches",
            "Manifest + dataset YAML",
            "Repeatable project structure",
        ],
        footer="Real life trigger: products move through a line and images are collected for AI inspection.",
        status_label="Input Ready",
        status_color=ACCENT,
    ),
    Scene(
        step=1,
        title="Data validation blocks bad training input",
        subtitle="Before training starts, the system checks manifest structure and missing file paths.",
        real_world=[
            "Broken paths waste compute and time",
            "Bad schemas create training failures later",
            "Validation should fail early, not silently",
        ],
        system_actions=[
            "Great Expectations checks required columns",
            "File existence checks verify images and optional labels",
            "A validation marker is written only when checks pass",
        ],
        outputs=[
            "reports/validation.ok",
            "Safer pipeline execution",
            "Early failure on bad data",
        ],
        footer="Quality begins before model training. Reliable MLOps starts with reliable inputs.",
        status_label="Validation Passed",
        status_color=SUCCESS,
    ),
    Scene(
        step=2,
        title="YOLOv8 trains and MLflow tracks every run",
        subtitle="Training produces weights, metrics, artifacts, and a reproducible experiment record.",
        real_world=[
            "Teams need repeatable experiments",
            "Different settings create different model quality",
            "Results must be searchable later",
        ],
        system_actions=[
            "YOLOv8 trains on the configured dataset",
            "MLflow logs parameters, metrics, and artifacts",
            "The best weights are packaged as an MLflow PyFunc model",
        ],
        outputs=[
            "runs/.../best.pt",
            "MLflow experiment history",
            "Portable serving model",
        ],
        footer="The project moves from raw data to a versioned model candidate with full experiment traceability.",
        status_label="Training Active",
        status_color=WARN,
    ),
    Scene(
        step=3,
        title="Evaluation gates and registry promotion protect production",
        subtitle="A model is evaluated, compared, and optionally promoted only if it meets quality policy.",
        real_world=[
            "Newer models are not always better",
            "Production should not update on hope alone",
            "Registry stages need simple promotion rules",
        ],
        system_actions=[
            "Validation metrics such as mAP@0.5 are extracted",
            "Quality gate can fail low-quality runs",
            "Champion vs challenger logic decides Production promotion",
        ],
        outputs=[
            "Model version tags",
            "Registry stage decision",
            "Controlled deployment path",
        ],
        footer="This is where training becomes platform engineering: the system decides if the model deserves production.",
        status_label="Gate Passed",
        status_color=SUCCESS,
    ),
    Scene(
        step=4,
        title="FastAPI serves the model through product APIs",
        subtitle="The model loads once at startup and becomes a usable application endpoint.",
        real_world=[
            "Operations teams need a stable inference service",
            "Apps and browsers need simple HTTP endpoints",
            "Security and observability matter in production",
        ],
        system_actions=[
            "FastAPI loads config and the predictor once",
            "Requests flow through auth, middleware, and structured logging",
            "The API exposes /predict, /health, /ready, /version, and /metrics",
        ],
        outputs=[
            "REST prediction API",
            "Health + readiness endpoints",
            "Traceable service logs",
        ],
        footer="This is the point where a trained model becomes a product surface that other systems can actually use.",
        status_label="Serving Live",
        status_color=ACCENT,
    ),
    Scene(
        step=5,
        title="Users inspect results in the browser UI",
        subtitle="An operator uploads an image, sees the JSON response, and views defect boxes on the preview.",
        real_world=[
            "Teams need a quick way to demo and inspect results",
            "Visual overlays help non-ML users trust predictions",
            "A lightweight interface is valuable for pilots and demos",
        ],
        system_actions=[
            "The frontend uploads the image to /predict",
            "Returned boxes and scores are drawn on a canvas overlay",
            "The raw JSON response stays visible for debugging and integration",
        ],
        outputs=[
            "Visual defect boxes",
            "Response payload preview",
            "Faster stakeholder demos",
        ],
        footer="The browser interface turns backend predictions into something operators and stakeholders can understand instantly.",
        status_label="UI Demo",
        status_color=ACCENT,
    ),
    Scene(
        step=6,
        title="Predictions are logged and drift is monitored over time",
        subtitle="The live system records behavior so teams can compare current production patterns against a baseline.",
        real_world=[
            "Production data changes after deployment",
            "Behavior drift can appear even when the API stays healthy",
            "Monitoring needs both system metrics and model signals",
        ],
        system_actions=[
            "Each prediction is appended to predictions.jsonl",
            "Prometheus scrapes runtime metrics and Grafana visualizes them",
            "Evidently compares reference vs current prediction behavior",
        ],
        outputs=[
            "JSONL prediction history",
            "Grafana dashboards",
            "HTML drift report",
        ],
        footer="Real-life operations continue after deployment: observe the service, observe the model, and retrain when needed.",
        status_label="Monitoring Active",
        status_color=SUCCESS,
    ),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a readable font. Fall back to Pillow's default if needed."""
    try:
        name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
        return ImageFont.truetype(name, size=size)
    except Exception:
        return ImageFont.load_default()


FONT_TITLE = _font(42, bold=True)
FONT_SUBTITLE = _font(22)
FONT_SECTION = _font(20, bold=True)
FONT_BODY = _font(18)
FONT_SMALL = _font(15)
FONT_BADGE = _font(16, bold=True)
FONT_STEP = _font(15, bold=True)


def _wrap(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont, width: int) -> list[str]:
    """Wrap text so long paragraphs fit inside cards without clipping."""
    words = text.split()
    if not words:
        return [""]

    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _draw_gradient(image: Image.Image) -> None:
    """Paint a subtle vertical background gradient for a more polished look."""
    px = image.load()
    for y in range(HEIGHT):
        mix = y / max(1, HEIGHT - 1)
        r = int(BG_TOP[0] * (1 - mix) + BG_BOTTOM[0] * mix)
        g = int(BG_TOP[1] * (1 - mix) + BG_BOTTOM[1] * mix)
        b = int(BG_TOP[2] * (1 - mix) + BG_BOTTOM[2] * mix)
        for x in range(WIDTH):
            px[x, y] = (r, g, b)


def _rounded(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill, outline, radius: int = 20, width: int = 2):
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


def _text_block(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    text: str,
    font: ImageFont.ImageFont,
    fill,
    width: int,
    line_gap: int = 6,
) -> int:
    """Draw wrapped text and return the next Y position."""
    lines = _wrap(draw, text, font, width)
    bbox = draw.textbbox((0, 0), "Ag", font=font)
    line_height = (bbox[3] - bbox[1]) + line_gap
    cy = y
    for line in lines:
        draw.text((x, cy), line, font=font, fill=fill)
        cy += line_height
    return cy


def _bullet_list(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    items: Iterable[str],
    width: int,
    bullet_color,
) -> int:
    """Draw a clean bullet list used across the three main cards."""
    cy = y
    for item in items:
        draw.ellipse((x, cy + 6, x + 8, cy + 14), fill=bullet_color)
        cy = _text_block(draw, x + 18, cy, item, FONT_BODY, TEXT, width - 18)
        cy += 8
    return cy


def _draw_steps(draw: ImageDraw.ImageDraw, active_index: int, pulse: bool) -> None:
    """Draw the lifecycle steps across the top with one current step highlighted."""
    steps = ["Capture", "Validate", "Train", "Promote", "Serve", "Inspect", "Monitor"]
    x = 60
    y = 112
    for index, step in enumerate(steps):
        w = 148
        h = 40
        active = index == active_index
        fill = ACCENT if active else PANEL_ALT
        outline = (155, 246, 236) if active and pulse else BORDER
        text_fill = (10, 20, 36) if active else MUTED
        _rounded(draw, (x, y, x + w, y + h), fill=fill, outline=outline, radius=18, width=3 if active else 2)
        label = f"{index + 1}. {step}"
        tw = draw.textlength(label, font=FONT_STEP)
        draw.text((x + (w - tw) / 2, y + 10), label, font=FONT_STEP, fill=text_fill)
        if index < len(steps) - 1:
            draw.line((x + w + 8, y + h / 2, x + w + 28, y + h / 2), fill=BORDER, width=4)
        x += 176


def _draw_badge(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, color) -> None:
    """Draw the top-right status badge for the active phase."""
    w = int(draw.textlength(text, font=FONT_BADGE)) + 34
    _rounded(draw, (x, y, x + w, y + 36), fill=color, outline=color, radius=16, width=2)
    draw.text((x + 16, y + 10), text, font=FONT_BADGE, fill=(10, 20, 36))


def _draw_browser_mock(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], scene: Scene, pulse: bool) -> None:
    """Draw a browser/API/dashboard style mock to make the workflow feel product-like."""
    x1, y1, x2, y2 = box
    _rounded(draw, box, fill=PANEL_ALT, outline=BORDER, radius=22, width=2)
    draw.rounded_rectangle((x1, y1, x2, y1 + 42), radius=22, fill=(13, 22, 40), outline=None)
    draw.ellipse((x1 + 16, y1 + 14, x1 + 26, y1 + 24), fill=DANGER)
    draw.ellipse((x1 + 32, y1 + 14, x1 + 42, y1 + 24), fill=WARN)
    draw.ellipse((x1 + 48, y1 + 14, x1 + 58, y1 + 24), fill=SUCCESS)
    draw.text((x1 + 84, y1 + 11), "defectguard.app", font=FONT_SMALL, fill=MUTED)

    inner = (x1 + 20, y1 + 62, x2 - 20, y2 - 20)
    draw.rounded_rectangle(inner, radius=18, fill=(10, 17, 31), outline=(33, 51, 86), width=1)
    draw.text((inner[0] + 16, inner[1] + 12), "Live product view", font=FONT_SECTION, fill=TEXT)

    if scene.step in (4, 5):
        # Browser UI style panel with an image and defect box.
        img_box = (inner[0] + 16, inner[1] + 48, inner[0] + 250, inner[1] + 240)
        draw.rounded_rectangle(img_box, radius=16, fill=(23, 37, 61), outline=(50, 77, 121), width=2)
        for stripe in range(5):
            yy = img_box[1] + 18 + stripe * 34
            draw.line((img_box[0] + 18, yy, img_box[2] - 18, yy), fill=(42, 58, 86), width=2)
        defect = (img_box[0] + 70, img_box[1] + 54, img_box[0] + 176, img_box[1] + 134)
        draw.rectangle(defect, outline=(155, 246, 236) if pulse else ACCENT, width=4)
        draw.rectangle((defect[0], defect[1] - 26, defect[0] + 126, defect[1] - 4), fill=(10, 17, 31))
        draw.text((defect[0] + 8, defect[1] - 22), "defect 97.2%", font=FONT_SMALL, fill=ACCENT)

        code_y = inner[1] + 56
        draw.text((inner[0] + 286, code_y), "{", font=FONT_BODY, fill=TEXT)
        draw.text((inner[0] + 302, code_y + 24), '"class_names": ["defect"],', font=FONT_SMALL, fill=MUTED)
        draw.text((inner[0] + 302, code_y + 46), '"scores": [0.972],', font=FONT_SMALL, fill=MUTED)
        draw.text((inner[0] + 302, code_y + 68), '"boxes": [[110, 84, 274, 206]]', font=FONT_SMALL, fill=MUTED)
        draw.text((inner[0] + 286, code_y + 92), "}", font=FONT_BODY, fill=TEXT)
    elif scene.step == 6:
        # Dashboard view.
        chart_a = (inner[0] + 16, inner[1] + 54, inner[0] + 260, inner[1] + 152)
        chart_b = (inner[0] + 16, inner[1] + 170, inner[0] + 260, inner[1] + 268)
        report = (inner[0] + 280, inner[1] + 54, inner[0] + 520, inner[1] + 268)
        for chart in (chart_a, chart_b, report):
            draw.rounded_rectangle(chart, radius=14, fill=(16, 25, 42), outline=(44, 70, 115), width=2)
        draw.text((chart_a[0] + 14, chart_a[1] + 10), "Request Rate", font=FONT_SMALL, fill=TEXT)
        draw.text((chart_b[0] + 14, chart_b[1] + 10), "Latency", font=FONT_SMALL, fill=TEXT)
        draw.text((report[0] + 14, report[1] + 10), "Drift Report", font=FONT_SMALL, fill=TEXT)
        pts = [(chart_a[0] + 18 + i * 32, chart_a[3] - 14 - v) for i, v in enumerate([20, 30, 18, 42, 38, 62, 74])]
        draw.line(pts, fill=ACCENT, width=4)
        pts = [(chart_b[0] + 18 + i * 32, chart_b[3] - 14 - v) for i, v in enumerate([40, 46, 44, 54, 52, 58, 70])]
        draw.line(pts, fill=WARN, width=4)
        draw.text((report[0] + 14, report[1] + 48), "Reference vs current", font=FONT_SMALL, fill=MUTED)
        draw.rounded_rectangle((report[0] + 14, report[1] + 86, report[2] - 14, report[1] + 126), radius=10, fill=(25, 55, 50))
        draw.text((report[0] + 28, report[1] + 98), "No severe service issues", font=FONT_SMALL, fill=SUCCESS)
        draw.rounded_rectangle((report[0] + 14, report[1] + 142, report[2] - 14, report[1] + 182), radius=10, fill=(68, 56, 20))
        draw.text((report[0] + 28, report[1] + 154), "Confidence drift detected", font=FONT_SMALL, fill=WARN)
    else:
        # Pipeline summary view for earlier scenes.
        cols = [
            ("Data", ["raw/", "manifest.csv", "dataset.yaml"]),
            ("Model", ["YOLOv8", "best.pt", "PyFunc"]),
            ("Ops", ["MLflow", "FastAPI", "Grafana"]),
        ]
        cx = inner[0] + 18
        for heading, rows in cols:
            panel = (cx, inner[1] + 56, cx + 160, inner[1] + 244)
            draw.rounded_rectangle(panel, radius=14, fill=(15, 24, 40), outline=(44, 70, 115), width=2)
            draw.text((panel[0] + 14, panel[1] + 14), heading, font=FONT_SECTION, fill=TEXT)
            ry = panel[1] + 54
            for row in rows:
                draw.rounded_rectangle((panel[0] + 14, ry, panel[2] - 14, ry + 32), radius=10, fill=(22, 35, 61))
                draw.text((panel[0] + 24, ry + 8), row, font=FONT_SMALL, fill=MUTED)
                ry += 42
            cx += 174


def _build_frame(scene: Scene, pulse: bool) -> Image.Image:
    image = Image.new("RGB", (WIDTH, HEIGHT), BG_TOP)
    _draw_gradient(image)
    draw = ImageDraw.Draw(image)

    # Header
    draw.text((60, 36), "DefectGuard Real-Life Workflow Simulation", font=FONT_TITLE, fill=TEXT)
    draw.text((60, 82), scene.title, font=FONT_SUBTITLE, fill=ACCENT)
    _draw_badge(draw, WIDTH - 220, 38, scene.status_label, scene.status_color)

    _draw_steps(draw, scene.step, pulse)

    # Three main storytelling panels
    left = (60, 176, 370, 610)
    center = (395, 176, 805, 610)
    right = (830, 176, 1140, 610)
    for box in (left, center, right):
        _rounded(draw, box, fill=PANEL, outline=BORDER, radius=24, width=2)

    draw.text((left[0] + 22, left[1] + 20), "Real-world context", font=FONT_SECTION, fill=TEXT)
    draw.text((center[0] + 22, center[1] + 20), "What the platform does", font=FONT_SECTION, fill=TEXT)
    draw.text((right[0] + 22, right[1] + 20), "What users get", font=FONT_SECTION, fill=TEXT)

    # Intro paragraph in center card.
    _text_block(draw, center[0] + 22, center[1] + 58, scene.subtitle, FONT_BODY, MUTED, 366)

    # Lists
    _bullet_list(draw, left[0] + 22, left[1] + 68, scene.real_world, 260, ACCENT)
    _bullet_list(draw, center[0] + 22, center[1] + 130, scene.system_actions, 366, SUCCESS)
    _bullet_list(draw, right[0] + 22, right[1] + 310, scene.outputs, 260, WARN)

    # Mock product area in the right card.
    _draw_browser_mock(draw, (right[0] + 18, right[1] + 58, right[2] - 18, right[1] + 286), scene, pulse)

    # Footer bar
    footer = (60, 632, WIDTH - 60, 688)
    _rounded(draw, footer, fill=PANEL_ALT, outline=BORDER, radius=20, width=2)
    _text_block(draw, footer[0] + 22, footer[1] + 16, scene.footer, FONT_BODY, TEXT, footer[2] - footer[0] - 44, line_gap=4)

    return image


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames: list[Image.Image] = []
    durations: list[int] = []

    # Create two frames per scene so the active step has a gentle pulse.
    for scene in SCENES:
        frames.append(_build_frame(scene, pulse=False))
        durations.append(950)
        frames.append(_build_frame(scene, pulse=True))
        durations.append(1150)

    # Pillow handles GIF encoding directly. optimize=True helps keep repo size reasonable.
    frames[0].save(
        OUT_PATH,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(f"Generated {OUT_PATH}")


if __name__ == "__main__":
    main()
