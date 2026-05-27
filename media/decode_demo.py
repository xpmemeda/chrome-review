from decode_heif import HeifDecoder
import time

if __name__ == "__main__":
    start_time = time.time()
    decoder = HeifDecoder("/home/tiger/local/libttheif/lib/shared/libttheif_dec.so")
    # img = decoder.decode("./image/image2.heif")
    img = decoder.decode("/home/tiger/workspace/resources/cat-632x1400.png.heic")
    # img = decoder.decode("/home/tiger/workspace/resources/shelf.heic")
    end_time = time.time()

    elapsed_ms = (end_time - start_time) * 1000  # 转换为毫秒
    print(f"函数耗时: {elapsed_ms:.3f} ms")

    output_png_path = r"./output.png"
    if img is not None:
        print(f"Image size: {img.size}")
        img.save(output_png_path)
        print(f"[INFO] Saved PNG to: {output_png_path}")
