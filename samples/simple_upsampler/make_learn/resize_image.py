from PIL import Image
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

            # 画像を２の倍数に合わせる
            width, height = img.size
            lr_w = width // 2
            lr_h = height // 2
            hr_w = lr_w * 2
            hr_h = lr_h * 2
            hr_img = img.crop( (0, 0, hr_w, hr_h) )

            # 画像を縮小する
            lr_img = hr_img.resize( (lr_w, lr_h) )

            # リサイズ後の画像を保存する
            filetitle = os.path.splitext(filename)[0]
            hr_img.save(os.path.join(hr_folder, filetitle + ".bmp"))
            lr_img.save(os.path.join(lr_folder, filetitle + ".bmp"))

        except:
            print( "error" )


