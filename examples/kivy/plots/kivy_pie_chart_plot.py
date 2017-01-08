#!/usr/bin/env python

"""
example plot for a pie chart in kivy
combination of codes:
pie_demo_features.py
kivy_pie_chart_plot.ply
"""

import matplotlib
matplotlib.use('module://kivy.garden.matplotlib.backend_kivy')
#matplotlib.use('Gtk')

import numpy as np
import matplotlib.pyplot as plt


def press(event):
    print('press released from test', event.x, event.y, event.button)


def release(event):
    print('release released from test', event.x, event.y, event.button)


def keypress(event):
    print('key down', event.key)


def keyup(event):
    print('key up', event.key)


def motionnotify(event):
    print('mouse move to ', event.x, event.y)


def resize(event):
    print('resize from mpl ', event)


def scroll(event):
    print('scroll event from mpl ', event.x, event.y, event.step)


def figure_enter(event):
    print('figure enter mpl')


def figure_leave(event):
    print('figure leaving mpl')


def close(event):
    print('closing figure')

N = 5
menMeans = (20, 35, 30, 35, 27)
menStd = (2, 3, 4, 1, 2)

ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

figure, ax = plt.subplots()

figure.canvas.mpl_connect('button_press_event', press)
figure.canvas.mpl_connect('button_release_event', release)
figure.canvas.mpl_connect('key_press_event', keypress)
figure.canvas.mpl_connect('key_release_event', keyup)
figure.canvas.mpl_connect('motion_notify_event', motionnotify)
figure.canvas.mpl_connect('resize_event', resize)
figure.canvas.mpl_connect('scroll_event', scroll)
figure.canvas.mpl_connect('figure_enter_event', figure_enter)
figure.canvas.mpl_connect('figure_leave_event', figure_leave)
figure.canvas.mpl_connect('close_event', close)

# The slices will be ordered and plotted counter-clockwise.
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
colors = ['yellowgreen', 'gold', 'lightskyblue', 'lightcoral']
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

plt.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=90)
# Set aspect ratio to be equal so that pie is drawn as a circle.
plt.axis('equal')

# fig = plt.figure()
fig = plt.gcf()
ax = fig.gca()
#####


def autolabel(rects):
    # attach some text labels
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2., 1.05 * height, '%d' %
                int(height), ha='center', va='bottom')

# autolabel(rects1)
# autolabel(rects2)


plt.draw()
#fig1.set_size_inches(18.5, 10.5, forward = True)
# plt.savefig("test.png")
plt.show()
