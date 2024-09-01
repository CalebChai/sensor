import socket
import time
import threading
import binascii
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque

class client_socket():
    def __init__(self):
        self.__ip       = "192.168.1.15"
        self.__port     = 9422
        self.__socket   = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.__cmd_stop = '43 AA 0D 0A'
        self.__cmd_auto = '48 AA 0D 0A'
        self.__cmd_ask  = '49 AA 0D 0A'
        self.start_time = 0
        self.record_thread = None
        self.draw_thread = None
        self.Fx = 0
        self.Fy = 0
        self.Fz = 0
        self.Mx = 0
        self.My = 0
        self.Mz = 0
        self.rc_time = 0
        self.time_list = deque(maxlen = 2000)
        self.Fx_list = deque(maxlen = 2000)
        self.Fy_list = deque(maxlen = 2000)
        self.Fz_list = deque(maxlen = 2000)
        self.Mx_list = deque(maxlen = 2000)
        self.My_list = deque(maxlen = 2000)
        self.Mz_list = deque(maxlen = 2000)
        self.Fx_bias = 0
        self.Fy_bias = 0
        self.Fz_bias = 0
        self.Mx_bias = 0
        self.My_bias = 0
        self.Mz_bias = 0
        self.calib = False
        self.line = [None]*6

    def __del__(self):
        self.stop_record()

    # Connect to the server
    def connect_server(self):
        try:
            self.__socket.connect((self.__ip, self.__port))
            print("Connected to the server")
            self.__socket.sendall(bytes.fromhex(self.__cmd_stop))
            time.sleep(0.005)
            self.start_time = time.time()
        except ConnectionRefusedError:
            print("Could not connect to the server")
        # finally:
        #     self.__socket.close()

    # ask & receive once
    def ask_data(self):
        while True: 
            self.__socket.sendall(bytes.fromhex(self.__cmd_ask))
            time.sleep(0.001)
            try:
                data = self.__socket.recv(24)
                if data:
                    hex_data=binascii.hexlify(data).decode()
                    list_hex=(','.join([hex_data[i+2:i+5] for i in range(0, len(hex_data)-8, 3)])).split(',')
                    list_int = [int(x,16) if int(x,16)<2048 else int(x,16)-4096 for x in list_hex]
                    self.Fx=list_int[0]*0.0537109375 - self.Fx_bias
                    self.Fy=list_int[1]*0.0537109375 - self.Fy_bias
                    self.Fz=list_int[2]*0.0537109375 - self.Fz_bias
                    self.Mx=list_int[3]*0.0078125 - self.Mx_bias
                    self.My=list_int[4]*0.0078125 - self.My_bias
                    self.Mz=list_int[5]*0.0078125 - self.Mz_bias
                    self.rc_time = time.time() - self.start_time
                    self.time_list.append(self.rc_time)
                    self.Fx_list.append(self.Fx)
                    self.Fy_list.append(self.Fy)
                    self.Fz_list.append(self.Fz)
                    self.Mx_list.append(self.Mx)
                    self.My_list.append(self.My)
                    self.Mz_list.append(self.Mz)
                    if(len(self.time_list) == 500 and self.calib == False):
                        self.Fx_bias = sum(self.Fx_list)/500
                        self.Fy_bias = sum(self.Fy_list)/500
                        self.Fz_bias = sum(self.Fz_list)/500
                        self.Mx_bias = sum(self.Mx_list)/500
                        self.My_bias = sum(self.My_list)/500
                        self.Mz_bias = sum(self.Mz_list)/500
                        self.calib = True
                        self.time_list.clear()
                        self.Fx_list.clear()
                        self.Fy_list.clear()
                        self.Fz_list.clear()
                        self.Mx_list.clear()
                        self.My_list.clear()
                        self.Mz_list.clear()
                        print("Calibration done!")
                    if(len(self.time_list) > 100 and self.calib == True):
                        with open('calib6.txt', 'a') as f:
                            f.write(str(self.rc_time) + ' ' + str(self.Fx) + ' ' + str(self.Fy) + ' ' + str(self.Fz) + ' ' + str(self.Mx) + ' ' + str(self.My) + ' ' + str(self.Mz) + '\n')
                        print("Time:",self.rc_time, "Fx:",self.Fx,"Fy:",self.Fy,"Fz:",self.Fz,"Mx:",self.Mx,"My ",self.My,"Mz:",self.Mz)
            except:
                print("no receive")

    def record_data(self):
        self.record_thread = threading.Thread(target = self.ask_data, name = 'record thread')
        self.record_thread.start()
        print("start record data!")

    def stop_record(self):
        if (self.record_thread is not None):
            self.record_thread.join()
        self.__socket.sendall(bytes.fromhex(self.__cmd_stop))
        self.__socket.close()
        time.sleep(0.005)
        
    # define a draw thread to draw Fx, Fy, Fz, Mx, My, Mz
    def animate(self, i):
        self.line[0].set_xdata(self.time_list)
        self.line[0].set_ydata(self.Fx_list)
        self.line[1].set_xdata(self.time_list)
        self.line[1].set_ydata(self.Fy_list)
        self.line[2].set_xdata(self.time_list)
        self.line[2].set_ydata(self.Fz_list)
        self.line[3].set_xdata(self.time_list)
        self.line[3].set_ydata(self.Mx_list)
        self.line[4].set_xdata(self.time_list)
        self.line[4].set_ydata(self.My_list)
        self.line[5].set_xdata(self.time_list)
        self.line[5].set_ydata(self.Mz_list)
        # update xlim and y lim of the plot
        plt.xlim(self.time_list[0], self.time_list[-1])
        plt.ylim(-10, 10)
        return self.line

    def draw_data(self):
        fig,ax = plt.subplots()
        self.line[0], = ax.plot(self.time_list, self.Fx_list, label='Fx')
        self.line[1], = ax.plot(self.time_list, self.Fy_list, label='Fy')
        self.line[2], = ax.plot(self.time_list, self.Fz_list, label='Fz')
        self.line[3], = ax.plot(self.time_list, self.Mx_list, label='Mx')
        self.line[4], = ax.plot(self.time_list, self.My_list, label='My')
        self.line[5], = ax.plot(self.time_list, self.Mz_list, label='Mz')

        ani = animation.FuncAnimation(fig, self.animate, interval=100)
        plt.legend()
        plt.show()

def main():
    myClient = client_socket()
    try:
        myClient.connect_server()
        myClient.record_data()
        myClient.draw_data()
    except:
        pass

if __name__ == "__main__":
    main()
