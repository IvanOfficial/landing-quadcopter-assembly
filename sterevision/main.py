from stereo_cam_class import double_cam
from tqdm import tqdm_notebook
name_left = "25797059"
name_right = "25791059"
d_c = double_cam(name_left, name_right)


a = 0
print("Input 'n'")
n = int(input())
for i in tqdm_notebook(range(n)):
    while (a != 1):
        d_c.show_me()
        a = int(input())
        if a == 1:
            print(i)
            d_c.save_photos(str(i), str(i))
    a = 0