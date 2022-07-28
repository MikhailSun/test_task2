import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import root_scalar
import matplotlib.pyplot as plt
import makecharts as mc
from copy import copy

#Исходные данные (все единицы в СИ, если не указано иное)

# расчет будем производить численно, разобъем расчетную область на N элементов вдоль трубы и M элементов поперек стенки трубы. Каждый такой элемент будет омываться маслом с одной стороны и водой с другой.
# слева и справа будет учитываться теплопроводность при передече энергии в соседние элементы трубы.
N=20
M=2

#для простоты предположим, что теплообменник - это одна труба находящаяся внутри другой. Во внутренней трубе течет горячее масло,
#во внешней - холодная вода. Теплообменник противоточный.
# У внешней трубы на внешней стенке задано условие адиабатности, т.е. отсутствует теплообмен. Весь теплообмен только с внутренней трубой.
L=10 #длина трубы (условного теплообменника)
d_oil=0.02 #диаметр внутренней трубы с маслом, м
d_water=0.04 #диаметр внешней трубы с маслом, м
h=0.002 #толщина стенки внутренней трубы, через которую происходит теплообмен, м
htc_oil=2000 #коэффициент теплобмена между маслом и стенкой (задаем произщвольно постоянным для простоты, но по-хорошему можно сделать его зависящим от условий течения в трубе), Вт/м2/К
htc_water=3000 #коэффициент теплобмена между водой и стенкой (задаем произщвольно постоянным для простоты, но по-хорошему можно сделать его зависящим от условий течения в трубе), Вт/м2/К
G_oil=0.5 #расход масла, кг/с
G_water=1 #расход воды, кг/с
Rho_oil=880 #плотность масла, кг/м3
Rho_water=1000 #плотность воды, кг/м3
Rho_tube=8920 #плотность меди, кг/м3
C_oil=1850 #теплоемкость масла, Дж/кг/К (в идеале можно сделать теплоемкость как функцию зависящую от температуры)
C_water=4200 #теплоемкость воды, Дж/кг/К (можно сделать как функцию)
C_tube=390 #уд.теплоемкость меди (предположим, что труба, через которую происходт теплообмен из меди), Дж/кг/К (можно сделать как функцию)
lambda_tube=395 #теплопроводность меди, Вт/м/К (можно сделать как функцию)

#временнЫе настройки расчета
T=100 #общее время расчета
time_oil_valve= {0: False,#время включения/выключения клапана по маслу
                 3:True,
                 45:False,
                 60:True,
                80:False}
time_water_valve= {0: False,  #время включения/выключения клапана по воде
                   5:True,
                    50:False,
                   73:True,
                    85:False}
t_oil_inlet= {0:30, #изменение температуры масла по времени
              5:40,
              25:50,
                50:60,
                75:80}
t_water_inlet= {0:25, #изменение температуры воды по времени
        40:25}

#Инициализация первоначальных условий:
#делаем допущение, что трубы изначально полностью заполнены. При открытии клапанов жидкости моментально приходят в движение с заданным расходом
t_oil_previous=[t_oil_inlet[0]]*N #задаем распределение температуры в трубе с маслом
t_water_previous=[t_water_inlet[0]]*N #задаем распределение температуры в трубе с водой
t_tube_previous=[[30.]*M]*N #задаем распределение температуры в материале трубы
oil_status=False #клапан по маслу закрыт
water_status=False #клапан по воде закрыт
t_oil_boundary=t_oil_inlet[0]
t_water_boundary=t_water_inlet[0]


#разные геометрические параметры:
L_element=L/N #длина элемента
h_element=h/M #высота элемента
F_tube_section = (2 * np.pi * (d_oil / 2 + h / 2) * h) #площадь поперечного сечения трубы
F_tube_external_surface = 2 * np.pi * (d_oil / 2 + h) * L_element #площадь внешней поверхности трубы с маслом
F_tube_internal_surface = 2 * np.pi * (d_oil / 2) * L_element #площадь внутренней поверхности трубы с маслом
dV_oil=(np.pi * (d_oil / 2)**2)*L_element #объем одного элемента внутри трубы с маслом
dV_water=((np.pi * (d_water / 2)**2) - (np.pi * (d_oil / 2+h)**2))*L_element #объем одного элемента внутри трубы с водой
#рассчитаем максимально допустимый шаг по времени исходя из того, что жидкость за этот шаг должна протечь расстояние не более одного элемента
V_oil=G_oil/Rho_oil/(np.pi*(d_oil/2)**2) #скорость масла в трубе
V_water=G_water/Rho_water/((np.pi*(d_water/2)**2)-(np.pi*(d_oil/2+h)**2)) #скорость масла в трубе
print(f'V_oil={V_oil}')
print(f'V_water={V_water}')
dt_oil=L_element/V_oil
dt_water=L_element/V_water
print(f'dt_oil={dt_oil}')
print(f'dt_water={dt_water}')
dt=min(dt_oil, dt_water)/2
print(f'dt={dt}')

#Массивы для записи результатов:
time=[0]
res_t_oil_in=[t_oil_inlet[0]]
res_t_oil_out=[t_oil_inlet[0]]
res_t_water_in=[t_water_inlet[0]]
res_t_water_out=[t_water_inlet[0]]
res_G_oil=[0]
res_G_water=[0]
res_t_oil_in_tube={77:np.nan,
                   20:np.nan} #температура масла внутри трубы вдоль трубы в заданный момент времени
res_t_water_in_tube={77:np.nan,
                     20:np.nan} #температура воды внутри трубы вдоль трубы в заданный момент времени

def find_t(t_assumed, t_oil, t_water, j, k):
    if k==0: #элемент контактирующий с маслом
        Q_up = htc_oil * (t_oil[j] - t_assumed) * F_tube_internal_surface * dt  # энергия от масла в трубу
    else:
        t_up = t_tube_previous[j][k-1]
        Q_up = lambda_tube * (t_up - t_assumed) / h_element * (2 * np.pi * (d_oil / 2 + k*h_element)) * dt  # энергия из верхнего элемента

    if k==M-1: #элемент контактирующий с водой
        Q_down = htc_water * (t_water[(N - 1) - j] - t_assumed) * F_tube_external_surface * dt  # энергия от трубы в воду
    else:
        t_down = t_tube_previous[j][k + 1]
        Q_down =lambda_tube * (t_down - t_assumed) / h_element * (2 * np.pi * (d_oil / 2 + (k+1)*h_element)) * dt  # энергия из верхнего элемента

    if j == 0:
        Q_left = 0.
    else:
        t_left = t_tube_previous[j - 1][k]
        Q_left = lambda_tube * (t_left - t_assumed) / L_element * F_tube_section * dt  # энергия в левый соседний элемент

    if j == N - 1:
        Q_right = 0.
    else:
        t_right = t_tube_previous[j + 1][k]
        Q_right = lambda_tube * (t_right - t_assumed) / L_element * F_tube_section * dt  # энергия в правый соседний элемент

    t_tube_new = (Q_up + Q_down + Q_left + Q_right) / (Rho_tube * L_element * F_tube_section * C_tube) + t_tube_previous[j][k]
    residual=t_assumed-t_tube_new
    return residual

#Расчет
Q_oil=[0.]*N
Q_water=[0.]*N
for i in range(1,int(T/dt)):
    t=i*dt
    time.append(t)
    if i%1000==0:
        print(f"step calculated: {i}")

    # проверяем статус клапана по маслу
    for t_, oil_status_ in time_oil_valve.items():
        if (abs(t_ - t) < dt * 0.5):
            oil_status = oil_status_
    for t_, t_oil_ in t_oil_inlet.items():
        if (abs(t_ - t) < dt * 0.5):
            t_oil_boundary = t_oil_
    if oil_status:
        G_oil_current=G_oil
        V_oil_current=V_oil
    else:
        G_oil_current = 0.
        V_oil_current = 0.

    # проверяем статус клапана по воде
    for t_, water_status_ in time_water_valve.items():
        if (abs(t_ - t) < dt * 0.5):
            water_status = water_status_
    for t_, t_water_ in t_water_inlet.items():
        if (abs(t_ - t) < dt * 0.5):
            t_water_boundary = t_water_
    if water_status:
        G_water_current = G_water
        V_water_current = V_water
    else:
        G_water_current = 0.
        V_water_current = 0.

    #1) расчет температур в трубе с маслом
    t_oil=[np.nan]*N #массив в котором будут храниться значения температуры масла вдоль трубы в данный момент времени
    dL = V_oil_current * dt  # расстояние протекаемой за один шаг по времени
    #цикл по элементам вдоль трубы c маслом
    for j in range(N):
        t_upstream=t_oil_previous[j-1] if j>0 else t_oil_boundary #температура элемента выше по потоку
        t_oil[j]=(dL/L_element)*t_upstream+(1-dL/L_element)*t_oil_previous[j]-Q_oil[j]/(Rho_oil*dV_oil*C_oil)
    t_oil_previous=t_oil

    #2) расчет температур в трубе с водой
    t_water=[np.nan]*N #массив в котором будут храниться значения температуры воды вдоль трубы в данный момент времени
    dL = V_water_current * dt  # расстояние протекаемой за один шаг по времени
    #цикл по элементам вдоль трубы c маслом
    for j in range(N):
        t_upstream=t_water_previous[j-1] if j>0 else t_water_boundary #температура элемента выше по потоку
        t_water[j]=(dL/L_element)*t_upstream+(1-dL/L_element)*t_water_previous[j]-Q_water[j]/(Rho_water*dV_water*C_water)
    t_water_previous=t_water


    #3) расчет теплопередачи в трубе
    t_tube = [[np.nan] * M]*N  # массив в котором будут храниться значения температуры воды вдоль трубы в данный момент времени
    # цикл по элементам вдоль трубы (труба моделируется толщиной в M элементов и в длину N элементов)
    for j in range(N):
        for k in range(M):
            t_assumed = t_tube_previous[j][k]
            t_tube[j][k] = root_scalar(find_t, x0=t_assumed, x1=t_assumed * 1.0001, method='secant',
                                    args=(t_oil, t_water, j, k)).root
            Q_oil[j] = htc_oil * (t_oil[j] - t_tube[j][k]) * F_tube_internal_surface * dt
            Q_water[j] = htc_water * (t_water[j] - t_tube[j][k]) * F_tube_external_surface * dt

    t_tube_previous=t_tube

    res_t_oil_in.append(t_oil_boundary)
    res_t_oil_out.append(t_oil[-1])
    res_t_water_in.append(t_water_boundary)
    res_t_water_out.append(t_water[-1])
    res_G_oil.append(G_oil_current)
    res_G_water.append(G_water_current)
    for t_ in res_t_oil_in_tube.keys():
        if (abs(t_ - t) < dt * 0.5):
            dist=[i*L_element for i in range(N)]
            res_t_oil_in_tube[t_] = [dist,t_oil]
    for t_ in res_t_water_in_tube.keys():
        if (abs(t_ - t) < dt * 0.5):
            dist = [i * L_element for i in range(N)]
            t_water_=copy(t_water)
            t_water_.reverse()
            res_t_water_in_tube[t_] = [dist,t_water_]
print(f"Calculating finished")

fig1=mc.Chart(points_for_plot=[{'x':time,'y':res_t_oil_in,'label':'t_oil_in','ls':'dashed','c':'red'},{'x':time,'y':res_t_oil_out,'label':'t_oil_out','c':'red'},
                               {'x':time,'y':res_t_water_in,'label':'t_water_in','ls':'dashed','c':'blue'},{'x':time,'y':res_t_water_out,'label':'t_water_out','c':'blue'}],xlabel='time, s',ylabel='temperature, C', dpi=150,figure_size=(5,5))
fig1.add_chart(points_for_plot=[{'x':time,'y':res_G_oil,'label':'G_oil','c':'red'},{'x':time,'y':res_G_water,'label':'G_water','c':'blue'}],xlabel='time, s',ylabel='Massflow, kg/s',)

print(res_t_water_in_tube)
for (time1,dist_temp_oil), (time2,dist_temp_water) in zip(res_t_oil_in_tube.items(), res_t_water_in_tube.items()):
    mc.Chart(points_for_plot=[{'x': dist_temp_oil[0], 'y':dist_temp_oil[1], 'label': f'Oil: time={time1}','c':'red'}, {'x': dist_temp_water[0], 'y':dist_temp_water[1], 'label': f'Water: time={time2}','c':'blue'}, ],
             xlabel='length of tube, m', ylabel='temperature, C',title=f'Time={time1}s', dpi=150, figure_size=(5, 5))

plt.show()