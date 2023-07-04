# %%
# /autofs/diamond2/share/corpus/DISFA/Videos_LeftCamera 以下にある avi ファイルの一覧を取得する
import os
import glob
import cv2
import numpy as np

# ファイル一覧を取得
files = glob.glob('/autofs/diamond2/share/corpus/DISFA/Videos_LeftCamera/*.avi')
# files = glob.glob('./*.mp4')
# ソートする
files.sort()


# %%
def proc(file, subject_name, logfile):
    # 出力用のディレクトリを作成
    outdir = f'./data/{subject_name}'    
    os.makedirs(outdir, exist_ok=True)

    # 顔検出器の初期化
    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    w_scale = 1.2 # 顔幅のスケール
    h_scale = 1.3 # 顔高さのスケール
    
    # ビデオファイルを1フレームずつ処理
    cap = cv2.VideoCapture(file)
    count = 0

    print(f"process start for {subject_name}", file=logfile, flush=True)
    while True:
        # 1フレーム読み込む
        ret, frame = cap.read()
        if not ret:
            print(f"end at count = {count}", file=logfile, flush=True)
            break

        faces = cascade.detectMultiScale(frame, 1.1, 3)
        if len(faces) == 0:
            print(f"could not found face at count = {count}", file=logfile, flush=True)
            continue
        elif len(faces) > 1:
            print(f"found multiple faces at count = {count}", file=logfile, flush=True)
            # 最も大きい顔を取得する
            d = []
            for x, y, w, h in faces:
                d.append(w * h)
            print(f"face size = {d}", file=logfile, flush=True)
            i = np.argmax(d)
            faces = faces[i:i+1]

        x1, y1, w, h = faces[0]
        center_x = x1 + w/2
        x1 = int(center_x - w/2 * w_scale)
        x2 = int(center_x + w/2 * w_scale)
        y1 = y1
        y2 = y1 + int(h * h_scale)

        cropped_image = frame[y1:y2, x1:x2]
        # 切り取った画像を適切にリサイズする
        if cropped_image.shape[0] < cropped_image.shape[1]:
            # 高さの方が短い場合は，幅を高さに合わせるが，中心がずれ内容にする
            diff = cropped_image.shape[1] - cropped_image.shape[0]
            cropped_image = cropped_image[:, diff//2:diff//2+cropped_image.shape[0]]
        else:
            # 幅の方が短い場合は，高さを幅に合わせる．この場合は，上から切り取る
            cropped_image = cropped_image[:cropped_image.shape[1], :]
        cropped_image = cv2.resize(cropped_image, (256, 256))        

        # 画像を保存
        cv2.imwrite(f'{outdir}/{count}.png', cropped_image)
        
        count += 1        


# %%
logfile = open('log.txt', 'w')

# ファイル名を表示
for file in files:
    # ファイル名は 'LeftVideoSN001_comp.avi' という形式．
    # そこから SN001 という文字列だけを取り出す．
    subject_name = os.path.basename(file)[9:14]

    proc(file, subject_name, logfile)
    # break

logfile.close()
    

# %%



