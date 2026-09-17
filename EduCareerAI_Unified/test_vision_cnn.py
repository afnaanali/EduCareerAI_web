import io
import numpy as np
from PIL import Image, ImageDraw

from core.dl_models import (
    predict_digit_cnn,
    preprocess_handwritten_digit,
    analyze_uploaded_marksheet
)

def run_tests():
    print("========================================")
    print("RUNNING FULL 7-STEP CNN VISION TEST SUITE")
    print("========================================")

    # ── TEST 1: Clear handwritten single digit (Digit 7 on white paper)
    img1 = Image.new("RGB", (120, 120), color="white")
    d1 = ImageDraw.Draw(img1)
    d1.line([25, 25, 95, 25], fill="black", width=8)
    d1.line([95, 25, 45, 95], fill="black", width=8)
    buf1 = io.BytesIO()
    img1.save(buf1, format="PNG")
    res1 = analyze_uploaded_marksheet(buf1.getvalue())
    print("TEST 1 (Digit 7 on paper):", res1["predicted_mark"], f"{res1['confidence']:.2f}%", "Status:", res1["status"])
    assert res1["success"] and res1["predicted_mark"] == "7"

    # ── TEST 2: Different handwritten digit (Digit 8 on white paper)
    img2 = Image.new("RGB", (120, 120), color="white")
    d2 = ImageDraw.Draw(img2)
    d2.ellipse([35, 20, 85, 55], outline="black", width=7)
    d2.ellipse([30, 50, 90, 100], outline="black", width=7)
    buf2 = io.BytesIO()
    img2.save(buf2, format="PNG")
    res2 = analyze_uploaded_marksheet(buf2.getvalue())
    print("TEST 2 (Digit 8 on paper):", res2["predicted_mark"], f"{res2['confidence']:.2f}%", "Status:", res2["status"])
    assert res2["success"] and res2["predicted_mark"] == "8"

    # ── TEST 3: MNIST-style image (White digit on black background)
    img3 = Image.new("RGB", (120, 120), color="black")
    d3 = ImageDraw.Draw(img3)
    d3.line([30, 30, 90, 30], fill="white", width=8)
    d3.line([90, 30, 45, 90], fill="white", width=8)
    buf3 = io.BytesIO()
    img3.save(buf3, format="PNG")
    res3 = analyze_uploaded_marksheet(buf3.getvalue())
    print("TEST 3 (MNIST style 7):", res3["predicted_mark"], f"{res3['confidence']:.2f}%")
    assert res3["success"] and res3["predicted_mark"] == "7"

    # ── TEST 4: Low-contrast photo style (grayish background)
    img4 = Image.new("RGB", (120, 120), color=(220, 220, 215))
    d4 = ImageDraw.Draw(img4)
    d4.line([30, 25, 90, 25], fill=(30, 30, 35), width=7)
    d4.line([90, 25, 45, 95], fill=(30, 30, 35), width=7)
    buf4 = io.BytesIO()
    img4.save(buf4, format="JPEG")
    res4 = analyze_uploaded_marksheet(buf4.getvalue())
    print("TEST 4 (Photo of paper 7):", res4["predicted_mark"], f"{res4['confidence']:.2f}%")
    assert res4["success"] and res4["predicted_mark"] == "7"

    # ── TEST 5: Invalid / Blank / Corrupted Input
    res5_blank = analyze_uploaded_marksheet(b"not an image")
    print("TEST 5a (Corrupted file error handling):", res5_blank["success"], "| Error:", res5_blank["error"])
    assert not res5_blank["success"]

    img5_blank = Image.new("RGB", (80, 80), color="white")
    buf5 = io.BytesIO()
    img5_blank.save(buf5, format="PNG")
    res5_white = analyze_uploaded_marksheet(buf5.getvalue())
    print("TEST 5b (Blank image error handling):", res5_white["success"], "| Error:", res5_white["error"])
    assert not res5_white["success"]

    # ── TEST 6: Existing Benchmark Sample functionality
    data_npz = np.load("data/mnist_benchmark_samples.npz")
    sample_2 = data_npz["2_sample_1"]
    res6 = predict_digit_cnn(sample_2)
    print("TEST 6 (Benchmark sample 2):", res6["predicted_digit"], f"{res6['confidence']:.2f}%")
    assert res6["predicted_digit"] == 2

    # ── TEST 7: Multi-digit segmentation test ('78')
    img7 = Image.new("RGB", (160, 100), color="white")
    d7 = ImageDraw.Draw(img7)
    d7.line([20, 20, 60, 20], fill="black", width=6)
    d7.line([60, 20, 35, 80], fill="black", width=6)
    d7.ellipse([90, 18, 140, 48], outline="black", width=6)
    d7.ellipse([85, 44, 145, 82], outline="black", width=6)
    buf7 = io.BytesIO()
    img7.save(buf7, format="PNG")
    res7 = analyze_uploaded_marksheet(buf7.getvalue())
    print("TEST 7 (Multi-digit '78'):", res7["predicted_mark"], "| Is multidigit:", res7["is_multidigit"], "| Conf:", f"{res7['confidence']:.2f}%")
    assert res7["success"]

    # ── TEST 8: Circled fractional mark test (e.g. circled 90 / 100 style)
    img8 = Image.new("RGB", (200, 200), color="white")
    d8 = ImageDraw.Draw(img8)
    # Teacher outer enclosing circle
    d8.ellipse([10, 10, 190, 190], outline="black", width=6)
    # Numerator (Digit 7 on top)
    d8.line([70, 35, 130, 35], fill="black", width=8)
    d8.line([130, 35, 85, 85], fill="black", width=8)
    # Horizontal fraction bar
    d8.line([40, 100, 160, 100], fill="black", width=8)
    # Denominator (Digit 1 on bottom: vertical line)
    d8.line([100, 115, 100, 175], fill="black", width=8)
    buf8 = io.BytesIO()
    img8.save(buf8, format="PNG")
    res8 = analyze_uploaded_marksheet(buf8.getvalue())
    print("TEST 8 (Circled Fraction '7 / 1'):", res8["predicted_mark"], "| Is fraction:", res8.get("is_fraction"), "| Conf:", f"{res8['confidence']:.2f}%")
    assert res8["success"] and res8["is_fraction"] and res8["predicted_mark"] == "7 / 1"

    print("========================================")
    print("ALL 8 VISION CNN TESTS COMPLETED & PASSED 100%!")
    print("========================================")


if __name__ == "__main__":
    run_tests()

