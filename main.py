# -*- coding:utf-8 -*-

"""
信号设计课程小组设计

@ by: Leaf
@ date: 2022-05-28
"""
import gr

import tkinter as tk
from tkinter import filedialog, Button, Label, Frame, ACTIVE, LEFT
from PIL import Image, ImageTk


class DisplayImage:
    """用于展示选择的图片"""
    def __init__(self, master):
        self.master = master
        master.title("GUI")
        self.Text_lab0 = Label(master, text='已加载图像/视频')
        self.Text_lab0.pack(pady=10)
        self.image_frame = Frame(master, bd=0, height=300, width=300, bg='white', highlightthickness=2,
                                 highlightbackground='gray', highlightcolor='black')
        self.image_frame.pack()

        self.Text_label = Label(master, text='加载待识别影像/视频')
        self.Text_label.place(x=60, y=410)
        self.Choose_image = Button(master, command=self.choose_img, text="图像",
                                   width=7, default=ACTIVE, borderwidth=0)
        self.Choose_image.place(x=50, y=450)
        self.Choose_image = Button(master, command=self.choose_video, text="视频",
                                   width=7, default=ACTIVE, borderwidth=0)
        self.Choose_image.place(x=120, y=450)
        self.Text_label2 = Label(master, text='运行手势识别程序')
        self.Text_label2.place(x=60, y=500)
        self.image_mosaic = Button(master, command=self.gesture_recognition, text="Gesture recognition",
                                   width=17, default=ACTIVE, borderwidth=0)
        self.image_mosaic.place(x=50, y=540)
        self.Text_label3 = Label(master, text='运行实时手势识别程序')
        self.Text_label3.place(x=300, y=410)
        self.realtime = Button(master, command=self.realtime_gr, text="Realtime\n gesture recognition",
                               width=17, height=6, default=ACTIVE, borderwidth=0)
        self.realtime.place(x=300, y=450)
        self.Text_label4 = Label(master, text='录入自定义手势')
        self.Text_label4.place(x=180, y=610)
        self.input = Button(master, command=self.input_image, text="Input gesture",
                            width=42, default=ACTIVE, borderwidth=0)
        self.input.place(x=60, y=650)

        self.gr = gr.Main()
        self.temp_dir = "temp"
        self.mode = 0
        self.directory = ""
        self.diy = 1

    def choose_img(self):
        self.mode = 1
        # 清空框架中的内容
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        self.directory = filedialog.askopenfilename()

        # 布局所选图片
        img = Image.open(self.directory).resize((300, 300))
        img.save(self.temp_dir + "/photo.png")
        image = ImageTk.PhotoImage(image=img)
        label = Label(self.image_frame, highlightthickness=0, borderwidth=0)
        label.configure(image=image)
        label.pack(side=LEFT, expand=True)

    def choose_video(self):
        # 清空框架中的内容
        self.mode = 2
        for widget in self.image_frame.winfo_children():
            widget.destroy()

        self.directory = filedialog.askopenfilename()

        # 布局所选图片
        img = Image.open(self.temp_dir+"/video.jpg").resize((300, 300))
        img.save(self.temp_dir + "/photo.png")
        image = ImageTk.PhotoImage(image=img)
        label = Label(self.image_frame, highlightthickness=0, borderwidth=0)
        label.configure(image=image)
        label.pack(side=LEFT, expand=True)

    def gesture_recognition(self):
        if self.mode == 1:
            self.gr.gr_img(self.directory, self.diy)
        elif self.mode == 2:
            self.gr.gr_video(self.directory, self.diy)

    def realtime_gr(self):
        self.gr.gr_realtime(self.diy)

    def input_image(self):
        self.diy = 1
        self.gr.ai_input()


def main():
    window = tk.Tk()
    DisplayImage(window)
    window.title('手势识别')
    window.geometry('500x720')
    window.mainloop()


if __name__ == '__main__':
    main()
