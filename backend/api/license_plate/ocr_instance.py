from paddleocr import PaddleOCR

# Instantiate once, reuse everywhere
ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en",
    show_log=False
)
