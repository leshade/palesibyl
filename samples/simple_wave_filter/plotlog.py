import os
import threading
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


log_filename = "trained_log.csv"


# ファイル監視イベントハンドラ
class EventHandler(FileSystemEventHandler):
	def on_any_event(self,e):
		global	log_filename
		if	os.path.basename(e.src_path) == log_filename:
			event.set()

event = threading.Event()
handler = EventHandler()
observer = Observer()
observer.schedule( handler, path=os.path.abspath("./") )
observer.start()


# CSV をグラフ表示
def PlotLogCSV():
	global	log_filename
	global	fig, ax

	try:
		data = pd.read_csv( log_filename )

	except:
		return

	tl = data.iloc[:,0]
	vl = data.iloc[:,1]

	if	len(tl) >= 2:
		x = list(range(len(tl)))

		ax.cla()

		yscale_log = False
		if	len(tl) >= 100:
			tlavg = np.mean( tl[len(tl)-50:] )
			lstd = max( np.std( tl[len(tl)-50:] ),
						np.std( vl[len(tl)-50:] ) )
			yscale_log = (tlavg + lstd * 5.0 < max(tl))

		if	yscale_log:
			plt.yscale( "log" )
		else:
			tvmin = min( min(tl), min(vl) )
			tvmax = max( max(tl), max(vl) )
			ax.set_ylim( min( tvmin - abs(tvmin) * 0.1, 0.0 ),
						 tvmax + abs(tvmax) * 0.2 )

		plt.plot( x, vl, color="red", label=data.columns[1] )
		plt.plot( x, tl, color="blue", label=data.columns[0] )

		plt.xlabel( "epoch" )
		plt.ylabel( "evaluation" )
		plt.legend(loc = 'upper right') 
		plt.grid()

		plt.draw()
		plt.pause( 0.3 )


# ウィンドウを閉じた時のハンドラ
def handle_close(evt):
	global is_closed
	is_closed = True

is_closed = False


fig, ax = plt.subplots()
fig.canvas.mpl_connect( "close_event", handle_close )

PlotLogCSV()


# ループ
while not is_closed:
	while not event.wait(0.1):
		plt.pause( 0.5 )
		if is_closed:
			break
	if is_closed:
		break

	event.clear()

	PlotLogCSV()

