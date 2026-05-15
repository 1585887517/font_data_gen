import os
import requests
from tqdm import tqdm


FONTS = [
    # ==================================================
    # 🚀 中文 - 基础印刷类 (SC = Simplified Chinese)
    # ==================================================
    {
        "name": "NotoSansSC-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/notosanssc/NotoSansSC%5Bwght%5D.ttf"
    },
    {
        "name": "NotoSerifSC-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/notoserifsc/NotoSerifSC%5Bwght%5D.ttf"
    },

    # ==================================================
    # 🚀 中文 - 楷体/手写风格 (解决识别歧义的关键)
    # ==================================================
    {
        "name": "MaShanZheng-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/mashanzheng/MaShanZheng-Regular.ttf"
    },
    {
        "name": "ZhiMangXing-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/zhimangxing/ZhiMangXing-Regular.ttf"
    },
    {
        "name": "ZCOOLXiaoWei-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/zcoolxiaowei/ZCOOLXiaoWei-Regular.ttf"
    },
    {
        "name": "LongCang-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/longcang/LongCang-Regular.ttf"
    },

    # ==================================================
    # 🚀 英文 - 常见印刷类
    # ==================================================
    {
        "name": "Roboto-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/apache/roboto/static/Roboto-Regular.ttf"
    },
    {
        "name": "PlayfairDisplay-Italic.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/playfairdisplay/PlayfairDisplay-Italic%5Bwght%5D.ttf"
    },
    {
        "name": "Montserrat-Regular.ttf",
        "url": "https://github.com/google/fonts/raw/main/ofl/montserrat/Montserrat%5Bwght%5D.ttf"
    }
]


def download_file(url, dest):
    try:
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        total = int(response.headers.get("content-length", 0))

        with open(dest, "wb") as f, tqdm(
            desc=dest,
            total=total,
            unit="B",
            unit_scale=True
        ) as bar:

            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

        print(f"✔ Downloaded: {dest}")

    except Exception as e:
        print(f"❌ Failed: {url}")
        print(e)


def main():

    out_dir = "fonts"
    os.makedirs(out_dir, exist_ok=True)

    for font in FONTS:
        path = os.path.join(out_dir, font["name"])
        download_file(font["url"], path)


if __name__ == "__main__":
    main()