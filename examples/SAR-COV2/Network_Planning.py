from pyhop import hop

# State: description of the current situation
# Task: description of an activity to perform

# Actions 

PTticket = 2.40

def walk(state,a,x,y):
    if state.loc[a] == x:
        state.loc[a] = y
        return state
    else: return False

def wait():
    pass

def drive(state,a,x,y):
    if state.loc[a] == x:
        state.loc[a] = y
        return state
    else: return False

def ride_bike(state,a,x,y):
    if state.loc[a] == x:
        state.loc[a] = y
        return state
    else: return False

def buy_ticket(state,a):
    if state.cash[a] >= PTticket:
        state.cash[a] = state.cash[a] - PTticket
        return state
    else: return False

def ride_subway(state, a, sx, sy):
    if state.loc[a] == sx and state.loc['subway'] == sx:
        state.loc['subway'] = sy
        state.loc[a] = sy
        return state
    else: return False

def ride_bus(state):
    if state.loc[a] == bx and state.loc['bus'] == bx:
        state.loc['bus'] = by
        state.loc[a] = by
        return state
    else: return False

def taxi_rate(dist):
    return (1.5 + 0.5 * dist)

def call_taxi(state,a,x):
    state.loc['taxi'] = x
    return state

def ride_taxi(state,a,x,y):
    if state.loc['taxi']==x and state.loc[a]==x:
        state.loc['taxi'] = y
        state.loc[a] = y
        state.owe[a] = taxi_rate(state.dist[x][y])
        return state
    else: return False

def pay_driver(state,a):
    if state.cash[a] >= state.owe[a]:
        state.cash[a] = state.cash[a] - state.owe[a]
        state.owe[a] = 0
        return state
    else: return False

hop.declare_operators(walk, drive, ride_bike, buy_ticket, ride_subway, ride_bus, call_taxi, ride_taxi, pay_driver)

# Method: parameterized description of a possible way to perform a compound task by performing a collection of subtasks

def travel_by_foot(state,a,x,y,sx,sy,bx,by,hc,hb):
    if state.dist[x][y] <= 1000: # Llamadas Heuristicas 
        return [('walk',a,x,y)]
    return False

def travel_by_car(state,a,x,y,sx,sy,bx,by,hc,hb):
    if state.car[a] == True:
        return [('drive',a,x,y)]
    return False

def travel_by_bike(state,a,x,y,sx,sy,bx,by,hc,hb):
    if state.bike[a] == True and state.dist[x][y] <= 2000:
        return [('ride_bike',a,x,y)]
    return False

def travel_by_subway(state,a,x,y,sx,sy,bx,by,hc,hb):
    if state.cash[a] >= PTticket:
        return[('walk',a,x,sx), ('buy_ticket',a), ('ride_subway',a,sx,sy), ('walk',a,sy,y)]

def travel_by_bus(state,a,x,y,sx,sy,bx,by,hc,hb):
    if state.cash[a] >= PTticket:
        return[('walk',a,x,bx), ('buy_ticket',a), ('ride_bus',a,bx,by), ('walk',a,by,y)]

def travel_by_taxi(state,a,x,y,sx,sy,bx,by,hc,hb):
    if state.cash[a] >= taxi_rate(state.dist[x][y]):
        return [('call_taxi',a,x), ('ride_taxi',a,x,y), ('pay_driver',a)]
    return False


hop.declare_methods('travel',travel_by_foot, travel_by_car, travel_by_bike, travel_by_subway, travel_by_bus, travel_by_taxi)


state1 = hop.State('state1') 
state1.loc = {'agent':'start'}  # Agent_id: position
state1.cash = {'agent':20}     # Agent_id: amount_Cash
state1.car = {'agent': False}   # Bool True or False has car random 30 %
state1.bike = {'agent': False}  # Bool Ture or False has bike random 15 %
state1.owe = {'agent':0}       # Agent_id: amount_owe
state1.dist = {'start':{'end':1500}, 'end':{'start':1500}}   # position:{dest:meters}, dest:po
#state1.time = {'start':{'end':20}, 'end':{'start':20}}
state1.loc_sub_entrance = {'subway_entrance': 'coord_sub_entrance'}
state1.loc_sub_exit = {'subway_exit': 'coord_sub_exit'}
# state1.loc_bus_entrance = {'bus_entrance': 'coord_bus_entrance'}
# state1.loc_bus_exit = {'bus_exit': 'coord_bus_exit'}


# Time variable 
# If new_transport_time < best_transport_so_far:
    # return state
# else:
    # return False

# Polution Variable

print("""
********************************************************************************
Call hop.plan(state1,[('travel','me','home','park')]) with different verbosity levels
********************************************************************************
""")

print("- If verbose=0 (the default), Pyhop returns the solution but prints nothing.\n")

hop.plan(state1,
         [('travel','agent','start','end','subway_entrance', 'subway_exit', 'bus_entrance', 'bus_exit', 'has_car', 'has_bike')],
         hop.get_operators(),
         hop.get_methods(),
         verbose = 1)