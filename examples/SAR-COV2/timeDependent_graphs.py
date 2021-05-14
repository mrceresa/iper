import csv
import time, os
import networkx as nx
import matplotlib as plt
import pickle 
import osmnx as ox
import math
start = time.time()

def time_function(self, current_time, interpolation_points):
    # Current time --> int representing the seconds from 00:00:00
    # interpolation_points --> array with touple (departure_time, travel_time)
    # Wait_time = departure_time - current_time; F = Wait_time + travel_time   
    next_departure = None
    for point in interpolation_points:
        if current_time <= point[0]:
            next_departure = point
        else:
            break
    if next_departure != None:
        y = (next_departure[0]-current_time) + next_departure[1]
        return y
    else: 
        print('No departure from this station after the current time' )
        return math.inf

class TimeDependentGraph():
    def __init__(self, direction, calmethod, date, day):
        ''' dir = directory path of the GTFS file set
            calmethod = 1--> use calendar.txt
                        = 2--> use calendar_dates.txt
                        = 3--> use both calendar and calendar_dates 
            day = string input (if calmethod = 1) for day of trip -> 'monday', 'tuesday', 'wednesday' ,'thursday', 'friday', 'saturday', 'sunday'
            date = string input (if calmethod = 2) on date as per GTFS-> YYYYMMDD
            validservices =  (dict to lookup valid service IDs running the specific day), {service_id:1} 
            validtrips = (dict to get trips valid that day-> maps to RouteID and route direction), {trip_id: [route_id, direction_id]}
            stopdata = (dict to get the name, lat and lon of each stop.)
            stoptrips = (dict of services (tripid, arrival, departure) serving a stop) stop:[[trip_id,arrival,departure]]
            getstopid = (dict of all the services = trip_id+stop_id: arrival, departure info),
        '''
        self.direction = direction
        self.calmethod = calmethod
        self.date = date
        self.day = day
        self.validservices = {}
        self.validtrips = {}
        self.stopdata = {}
        self.stoptrips = {}
        self.getstopid = {}
        self.stop_times = []
        self.G = nx.MultiDiGraph(crs = 'EPSG:32631')

    def get_valid_day_calendar(self):
        if calmethod == 1:
            #1.a) get calendar for service id and valid day
            with open(self.direction+'/calendar.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                fields = next(reader)
                serviceday = dict(zip(fields, range(len(fields))))
                for row in reader:
                    if row[serviceday[self.day]] == '1':
                        self.validservices[row[serviceday['service_id']]] = 1

        elif calmethod == 2:
            #1.b) Valid services based on calendar_dates.txt
            with open(self.direction+'/calendar_dates.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                next(reader)
                for row in reader:
                    if row[1] == self.date and row[2] == '1':
                        self.validservices[row[0]] = 1

        elif calmethod == 3:
            #1.c) get calendar for service id and valid day
            with open(self.direction+'/calendar.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                fields = next(reader)
                serviceday = dict(zip(fields, range(len(fields))))
                for row in reader:
                    if row[serviceday[self.day]] == '1':
                        self.validservices[row[serviceday['service_id']]] = 1

            with open(self.direction+'/calendar_dates.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                next(reader)
                serviceexcept = {}
                for row in reader:
                    if row[1] == self.date:
                        serviceexcept[row[0]] = row[2] #1 -> add, 2 --> remove

            #filter out the services not applicable to a date...
            for key in serviceexcept.keys():
                if serviceexcept[key] == '1':   #1 -> add, 2 --> remove
                    if key not in self.validservices:
                        self.validservices[key] = 1  # -> insert an added service if it does not already exist
                elif serviceexcept[key] == '2':
                    if key in self.validservices:
                        self.validservices.pop(key)  # --> pop out an invalid service for the day

    def get_valid_trips(self):
        with open(self.direction+'/trips.txt', 'r') as fn:
            reader = csv.reader(fn, delimiter=',')
            fields = next(reader)
            attix = dict(zip(fields, range(len(fields))))
            for row in reader:
                if row[attix['service_id']] in self.validservices:
                    self.validtrips[row[attix['trip_id']]]=[row[attix['route_id']], row[attix['direction_id']]]

    def get_stop_data(self):
        with open(self.direction+'/stops.txt', 'r', encoding='utf-8-sig') as fn:
            reader = csv.reader(fn, delimiter=',')
            fields = next(reader)
            attix = dict(zip(fields, range(len(fields))))
            for row in reader:
                self.stopdata[row[attix['stop_id']]] = [row[attix['stop_name']], float(row[attix['stop_lat']]), float(row[attix['stop_lon']])]

    def build_valid_journeys(self):
        with open(self.direction+'/stop_times.txt', 'r') as fn:
            reader = csv.reader(fn, delimiter=',')
            fields = next(reader)
            attix = dict(zip(fields, range(len(fields))))
            cnt=0
            for row in reader:
                #this builds stop to stop times for valid trips...
                if row[attix['trip_id']] in self.validtrips:
                    self.stop_times.append(row)
                    self.getstopid[row[attix['trip_id']]+'^'+row[attix['stop_id']]]=[row[attix['stop_id']], row[attix['arrival_time']], row[attix['departure_time']]]
                #this builds wihin stop transfers...
                    try:
                        if row[attix['stop_id']] in self.stoptrips:
                            arr = list(map(int, row[attix['arrival_time']].split(':')))
                            arrtm = arr[0]*3600 + arr[1]*60 + arr[2]
                            dep = list(map(int, row[attix['departure_time']].split(':')))
                            deptm = dep[0]*3600 + dep[1]*60 + dep[2]
                            self.stoptrips[row[attix['stop_id']]].append([row[attix['trip_id']], arrtm, deptm])
                        else:
                            arr = list(map(int, row[attix['arrival_time']].split(':')))
                            arrtm = arr[0]*3600 + arr[1]*60 + arr[2]
                            dep = list(map(int, row[attix['departure_time']].split(':')))
                            deptm = dep[0]*3600 + dep[1]*60 + dep[2]
                            self.stoptrips[row[attix['stop_id']]]=[[row[attix['trip_id']], arrtm, deptm]]
                    except:
                        if cnt < 10:
                            print('Failed to convert data for row: ', row)
                            cnt +=1
                        else:
                            pass

    def built_graph(self, edges):
        self.G.add_weighted_edges_from(edges, 'time')

    def add_coord_to_nodes(self):
        for node_id in self.G.nodes:
            try:
                route, station = node_id.split('^')
                stop_lat = self.stopdata[station][1]
                stop_lon = self.stopdata[station][2]
                self.G.nodes[node_id]['x'] = stop_lon #+ uniform(0,0.001)
                self.G.nodes[node_id]['y'] = stop_lat #+ uniform(0,0.003)
                if route == 'Super':
                    self.G.nodes[node_id]['x'] = stop_lon + 0.002
                    self.G.nodes[node_id]['y'] = stop_lat + 0.000
            except:
                print('Failed to find the station id: ', station)

    def plot_graph(self, ax=None, figsize=(8, 8), bgcolor="#111111", node_color="w", node_size=15, node_alpha=None, node_edgecolor="none", node_zorder=1, edge_color="#999999", edge_linewidth=1, edge_alpha=None, show=True, close=False, save=False, filepath=None, dpi=300, bbox=None):
        fig, ax = ox.plot_graph(self.G, ax=ax, figsize=figsize, bgcolor=bgcolor, node_color=node_color, node_size=node_size, node_alpha=node_alpha, node_edgecolor=node_edgecolor, node_zorder=node_zorder, edge_color=edge_color, edge_linewidth=edge_linewidth, edge_alpha=edge_alpha, show=show, close=close, save=save, filepath=filepath, dpi=dpi, bbox=bbox)
        return fig, ax

class TramTimeDependentGraph(TimeDependentGraph):
    def __init__(self, direction, calmethod, date, day):
        t0 = time.time()
        super().__init__(direction, calmethod, date, day)

        # Get the calendar for the valid day 
        self.get_valid_day_calendar()
        # Get valid trip_id based on valid services
        self.get_valid_trips()
        #Get Stop Data
        self.get_stop_data()
        # Build the journeys from valid trips
        self.build_valid_journeys()
        #Finsih lookup tables

        # Built Edges
        self.edges = self.built_edges()
        #Built Graph
        self.built_graph(self.edges)
        self.add_coord_to_nodes()

        print('Finished building Tram graph in: ', time.time()-t0, ' secs')

        self.plot_graph()     

    def filter_timetable_by_route(self, stop_id, route):
        filtered_times = []
        all_times = self.stoptrips[str(stop_id)]
        for event in all_times:
            if self.validtrips[event[0]] == route:  # Checks the route and the direction 
                filtered_times.append(event)
        return filtered_times

    def built_edges(self):
        route_built = []
        edges = []
        transfer_time = 120
        for trip in self.validtrips.items():
            # Check which routes have we already built
            route = trip[1]
            if route in route_built:
                continue
            else:
                # Build the linial graph: Each one of the routes
                route_stops = {} 
                for stop_service in self.stop_times: #Stop Times are the all stops times for all valid trips
                    if stop_service[0] == trip[0]: # Check if the trips belongs to the route.
                        route_stops[stop_service[3]] = stop_service[4]     #dict (stop_id: stop_sec)
                #route_stops = dict(sorted(route_stops.items(), key=lambda item: item[1]))   
                #print(route_stops)
                
                #Build the interpolation points and the transfers edges.
                stop_ids = list(route_stops.keys())
                for stop_index in range(0, len(stop_ids)):
                    stop_id_departure = stop_ids[stop_index]
                    if stop_id_departure == stop_ids[-1]:
                        edges.append([str(route[0])+'_'+str(route[1])+'^'+stop_id_departure,
                                    'Super'+'^'+stop_id_departure,
                                    transfer_time])
                        edges.append(['Super'+'^'+stop_id_departure,
                                    str(route[0])+'_'+str(route[1])+'^'+stop_id_departure,
                                    transfer_time])
                        break
                    else:
                        stop_id_arrival = stop_ids[stop_index+1]
                        #Linear
                        #Get the interpolation points 
                        dept_events = self.filter_timetable_by_route(stop_id_departure, route)
                        arri_events = self.filter_timetable_by_route(stop_id_arrival, route)
                        interpolation_points = []
                        for dept, arri in zip(dept_events, arri_events):
                            interpolation_points.append([dept[2], arri[1] - dept[2]])   #Departure time, travel time             
                        
                        #Append edge
                        edges.append([str(route[0])+'_'+str(route[1])+'^'+stop_id_departure,
                                    str(route[0])+'_'+str(route[1])+'^'+stop_id_arrival,
                                    interpolation_points])
                        #Transfer
                        edges.append([str(route[0])+'_'+str(route[1])+'^'+stop_id_departure,
                                    'Super'+'^'+stop_id_departure, 
                                    transfer_time])
                        edges.append(['Super'+'^'+stop_id_departure,
                                    str(route[0])+'_'+str(route[1])+'^'+stop_id_departure,
                                    transfer_time])

                route_built.append(route)

        print('The Graph has ' + str(len(edges)) + ' edges')
        return edges

class TMBTimeDepenedentGraph(TimeDependentGraph):
    def __init__(self):
        t0 = time.time()
        super().__init__(direction, calmethod, date, day)

        # Get the calendar for the valid day 
        self.get_valid_day_calendar()
        # Get valid trip_id based on valid services
        self.get_valid_trips()
        #Get Stop Data
        self.get_stop_data()
        # Build the journeys from valid trips
        self.build_valid_journeys()
        #Finsih lookup tables

path = os.getcwd()
dirlocBaixLlobregat = path + '/GTFS/TBX'         #Dir where the GTFS txt files are.
dirlocBesos = path + '/GTFS/TBS'
dirlocprova = path + '/GTFS/Prova'
saveloc = path + '/GTFS/Graphs/Prova'
calmethod = 3
date = '20210326' #'20150611'
day = 'friday'
save = False

G_tram_BaixLlobregat = TramTimeDependentGraph(dirlocBaixLlobregat, calmethod, date, day)
G_tram_Besos = TramTimeDependentGraph(dirlocBesos, calmethod, date, day)
G_prova = TramTimeDependentGraph(dirlocprova, calmethod, date, day)

if save == True:
    # Dump graph
    with open(saveloc + "/" + date + ".p", 'wb') as f:
        pickle.dump(G, f)