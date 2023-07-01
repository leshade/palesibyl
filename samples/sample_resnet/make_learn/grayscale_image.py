from PIL import Image
from PIL import ImageEnhance
import os

src_folder = "src"
lr_folder = "..\\learn\\source"
hr_folder = "..\\learn\\teacher"

for filename in os.listdir(src_folder):

    # ファイルが画像ファイルであるかを確認する
    lfn = filename.lower()
    if lfn.endswith(".png") or lfn.endswith(".jpg") or lfn.endswith(".bmp"):

        # 画像ファイルを開く
        print( filename, "..." )
        try:
            img = Image.open(os.path.join(src_folder, filename))

            # 画像を狭いほうを 512 以下、かつ16の倍数に合わせる
            width, height = img.size
            lr_w = width
            lr_h = height
            while	(lr_w > 512) and (lr_h > 512):
                lr_w = lr_w // 2
                lr_h = lr_h // 2

            img = img.resize( (lr_w, lr_h) )

            lr_w = lr_w // 16
            lr_h = lr_h // 16
            hr_w = lr_w * 16
            hr_h = lr_h * 16
            hr_img = img.crop( (0, 0, hr_w, hr_h) )
            print( width, "x", height, " -> ", hr_w, "x", hr_h )

            # グレイスケール画像を生成する
            lr_img = hr_img.convert( "L" )
            lr_img = lr_img.convert( "RGB" )

            # 学習しやすいようにカラー画像の彩度を上げる
            sat = ImageEnhance.Color( hr_img )
            hr_img = sat.enhance( 2.0 )

            # リサイズ後の画像を保存する
            filetitle = os.path.splitext(filename)[0]
            hr_img.save(os.path.join(hr_folder, filetitle + ".bmp"))
            lr_img.save(os.path.join(lr_folder, filetitle + ".bmp"))

        except:
            print( "error" )


