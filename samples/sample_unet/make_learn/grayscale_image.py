from PIL import Image
from PIL import ImageEnhance
import os

src_folder = "src"
lr_folder = "..\\learn\\source"
hr_folder = "..\\learn\\teacher"
dst_size = 320

for filename in os.listdir(src_folder):

    # ファイルが画像ファイルであるかを確認する
    lfn = filename.lower()
    if lfn.endswith(".png") or lfn.endswith(".jpg") or lfn.endswith(".bmp"):

        # 画像ファイルを開く
        print( filename, "..." )
        try:
            img = Image.open(os.path.join(src_folder, filename))

            # 画像を狭いほうを 320 に合わせる
            s_width, s_height = img.size
            width, height = img.size
            lr_w = width
            lr_h = height
            if	width < height:
            	height = height * dst_size // width
            	width = dst_size
            else:
            	width = width * dst_size // height
            	height = dst_size

            img = img.resize( (width, height) )

            # 320x320 に切り出す
            hr_img = img.crop( ((width - dst_size)//2,
            					(height - dst_size)//2,
            					dst_size, dst_size) )
            width = dst_size
            height = dst_size

            # 画像を16の倍数に合わせる
            lr_w = width // 16
            lr_h = height // 16
            hr_w = lr_w * 16
            hr_h = lr_h * 16
            hr_img = img.crop( (0, 0, hr_w, hr_h) )
            print( s_width, "x", s_height, " -> ", hr_w, "x", hr_h )

            # グレイスケール画像を生成する
            lr_img = hr_img.convert( "L" )
            lr_img = lr_img.convert( "RGB" )

            # 分類器（エンコーダー）学習用にカラー画像の彩度を上げる
            sat = ImageEnhance.Color( hr_img )
            hr_img = sat.enhance( 2.0 )

            # リサイズ後の画像を保存する
            filetitle = os.path.splitext(filename)[0]
            hr_img.save(os.path.join(hr_folder, filetitle + ".bmp"))
            lr_img.save(os.path.join(lr_folder, filetitle + ".bmp"))

        except:
            print( "error" )


