from tkinter import *
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib

root = Tk()
frame = Frame(root, height=600,width = 800,bg="red")
frame.pack()

E1 = Entry(frame, bd =5,width=100)

E1.pack(side = LEFT)

bluebutton = Button(frame, text="Search", fg="blue")
bluebutton.pack( side = RIGHT )

labelframe = LabelFrame(root, text="This is a LabelFrame",height=600,width = 800)
labelframe.pack(fill="both", expand="yes",side=LEFT)

left = Label(labelframe, text="Inside the LabelFrame")
left.pack()

scrollbar = Scrollbar(labelframe)
scrollbar.pack( side = RIGHT, fill=Y )

text = Text(labelframe, yscrollcommand = scrollbar.set)
# text.insert(INSERT, "Hello.....")
# text.insert(END, "Bye Bye.....")
text.pack( fill = BOTH)




labelframe2 = LabelFrame(root, text="This is a LabelFrame",height=600,width = 800)
labelframe2.pack(fill="both", expand="yes",side=RIGHT)

left2 = Label(labelframe2, text="Graph")
left2.pack()

# ---------------------------------------------------------------------------------------------------------------------

# style.use('ggplot')

matplotlib.use('TkAgg')

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)

def animate(i):
    pullData = open('twitter-output.txt', 'r').read()
    lines = pullData.split('\n')


    xs = []
    ys = []

    x = 0
    y = 0

    for line in lines:
        x += 1
        if "pos" in line:
            y += 1
        elif "neg" in line:
            y -= 0.3

        xs.append(x)
        ys.append(y)


    ax1.clear()
    ax1.plot(xs, ys)

ani = animation.FuncAnimation(fig, animate, interval = 1000)
plt.show()

canvas = FigureCanvasTkAgg(master=labelframe2,figure=fig)
canvas.get_tk_widget().pack()
canvas.draw()


# ---------------------------------------------------------------------------------------------------------------------

root.mainloop()
