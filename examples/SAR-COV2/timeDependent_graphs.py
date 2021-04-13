import csv
import time, os
import networkx as nx
import matplotlib as plt
import pickle 
import osmnx as ox
start = time.time()

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

    def time_function(self, current_time, interpolation_points):
        # Current time is an int representing the seconds from 00:00:00
        # array with touple (departure_time, travel_time)
        next_departure = None
        for point in interpolation_points:
            if current_time <= point[0]:
                next_departure = (point)
            else:
                break
        y = (next_departure[0]-current_time) + next_departure[1]
        return y

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
                print(route_stops)
                
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

        print(len(edges))
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
dirloc = path + '/GTFS/TBX'
saveloc = path + '/GTFS/Graphs/Prova'
calmethod = 3
date = '20210326' #'20150611'
day = 'friday'
print(TramTimeDependentGraph(dirloc, calmethod, date, day))
exit()


def time_weight():
    pass
    #piecewise linear functions
    #add an in- terpolation point p := (τ1, ∆(τ1, τ2)) to the function f that belongs to the edge between S1 and S2

def BuildSearchGraph(dir, xferpen, calmethod, date='', day=''):
    '''dir = directory path of the GTFS file set
       xferpen = transfer penalty in seconds - needed for getting fewer transfer in paths
       calmethod = 1--> use calendar.txt
                 = 2--> use calendar_dates.txt
                 = 3--> use both calendar and calendar_dates 
       day = string input (if calmethod = 1) for day of trip -> 'monday', 'tuesday', 'wednesday' ,'thursday', 'friday', 'saturday', 'sunday'
       date = string input (if calmethod = 2) on date as per GTFS-> YYYYMMDD
       The function returns: 1) edges (list to build Graph in networkX),
                             2) validservices (dict to lookup valid service IDs running the specific day), {service_id:1} 
                             3) validtrips (dict to get trips valid that day-> maps to RouteID), {trip_id, route_id}
                             4) stoptrips (dict of services serving a stop),
                             5) getstopid (dict of trip_id+stop_id -> arrival, departure info),
                             6) stopdata (dict of stop properties)
    '''
    t0 = time.time()
    filesindir = set(os.listdir(dir))
    print('Files found in directory: ', filesindir)
    reqdfiles = ['stops.txt', 'trips.txt', 'stop_times.txt']#, 'transfers.txt']
    if calmethod == 1:
        reqdfiles.append('calendar.txt')
    elif calmethod == 2:
        reqdfiles.append('calendar_dates.txt')
    elif calmethod == 3:
        reqdfiles.append('calendar.txt')
        reqdfiles.append('calendar_dates.txt')
        
    filesok = 0
    if filesindir.issuperset(reqdfiles):
        filesok=1
        print('Starting to process feed data...')
        if calmethod == 1:
            #1.a) get calendar for service id and valid day
            with open(dir+'/calendar.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                fields = next(reader)
                serviceday = dict(zip(fields, range(len(fields))))
                #service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
                validservices = {}
                for row in reader:
                    if row[serviceday[day]] == '1':
                        validservices[row[serviceday['service_id']]] = 1

        elif calmethod == 2:
            #1.b) Valid services based on calendar_dates.txt
            with open(dir+'/calendar_dates.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                # extracting field names through first row
                next(reader)
                #service_id,date,exception_type
                validservices = {}
                for row in reader:
                    if row[1] == date and row[2] == '1':
                        validservices[row[0]] = 1
            
            #print(validservices)

        elif calmethod == 3:
            #1.c) get calendar for service id and valid day
            with open(dir+'/calendar.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                fields = next(reader)
                serviceday = dict(zip(fields, range(len(fields))))
                #service_id,monday,tuesday,wednesday,thursday,friday,saturday,sunday,start_date,end_date
                validservices = {}
                for row in reader:
                    if row[serviceday[day]] == '1':
                        validservices[row[serviceday['service_id']]] = 1

            with open(dir+'/calendar_dates.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                next(reader)
                #service_id,date,exception_type
                serviceexcept = {}
                for row in reader:
                    if row[1] == date:
                        serviceexcept[row[0]] = row[2] #1 -> add, 2 --> remove

            #valid services has all services without exception... now we filter out the ones not applicable to a date...
            for key in serviceexcept.keys():
                if serviceexcept[key] == '1':   #1 -> add, 2 --> remove
                    if key not in validservices:
                        validservices[key] = 1  # -> insert an added service if it does not already exist
                elif serviceexcept[key] == '2':
                    if key in validservices:
                        validservices.pop(key)  # --> pop out an invalid service for the day

        #2) get valid trip_id based on valid services
        with open(dir+'/trips.txt', 'r') as fn:
            reader = csv.reader(fn, delimiter=',')
            # extracting field names through first row
            fields = next(reader)
            attix = dict(zip(fields, range(len(fields))))
            #print(attix)
            validtrips = {} #gives route id based on trip id
            for row in reader:
                if row[attix['service_id']] in validservices:
                    validtrips[row[attix['trip_id']]]=row[attix['route_id']]

        #print(validtrips)

        #3) now build the journeys from valid trips
        with open(dir+'/stop_times.txt', 'r') as fn:
            reader = csv.reader(fn, delimiter=',')
            # extracting field names through first row
            fields = next(reader)
            attix = dict(zip(fields, range(len(fields))))
            #print(attix)
            stoptrips = {}
            getstopid = {}
            edges = []
            stop_times = []
            cnt=0
            for row in reader:
                #this builds stop to stop times for valid trips...
                #trip_id,arrival_time,departure_time,stop_id,stop_sequence,stop_headsign,pickup_type,drop_off_type,shape_dist_traveled
                if row[attix['trip_id']] in validtrips:
                    stop_times.append(row)
                    getstopid[row[attix['trip_id']]+'^'+row[attix['stop_id']]]=[row[attix['stop_id']], row[attix['arrival_time']], row[attix['departure_time']]]
                #this builds wihin stop transfers...
                #MANY BUSSES HAVE NOT TIMES. I SHOULD INTERPOLATE TO GET THE TIMES AND PLACE THEM BACK INTO THE FEED
                    try:
                        if row[attix['stop_id']] in stoptrips:
                            arr = list(map(int, row[attix['arrival_time']].split(':')))
                            arrtm = arr[0]*3600 + arr[1]*60 + arr[2]
                            dep = list(map(int, row[attix['departure_time']].split(':')))
                            deptm = dep[0]*3600 + dep[1]*60 + dep[2]
                            stoptrips[row[attix['stop_id']]].append([row[attix['trip_id']], arrtm, deptm])
                        else:
                            arr = list(map(int, row[attix['arrival_time']].split(':')))
                            arrtm = arr[0]*3600 + arr[1]*60 + arr[2]
                            dep = list(map(int, row[attix['departure_time']].split(':')))
                            deptm = dep[0]*3600 + dep[1]*60 + dep[2]
                            stoptrips[row[attix['stop_id']]]=[[row[attix['trip_id']], arrtm, deptm]]
                    except:
                        if cnt < 10:
                            print('Failed to convert data for row: ', row)
                            cnt +=1
                        else:
                            pass
            
            stop_times.sort(key=lambda k: (k[0], k[4]))
            #print(cnt)                    
            
        print('Finished building basic lookups and filtering data: ', time.time() - start, ' secs')
       
        start1 = time.time()
        cnt = 0
        for i in range(0, len(stop_times)-1):
            if stop_times[i][attix['trip_id']] == stop_times[i+1][attix['trip_id']]: #Check stop sequence is are truly concadenated. 
                try:
                    arr = list(map(int, stop_times[i+1][attix['arrival_time']].split(':')))
                    arrtm = arr[0]*3600 + arr[1]*60 + arr[2]
                    dep = list(map(int, stop_times[i][attix['departure_time']].split(':')))
                    deptm = dep[0]*3600 + dep[1]*60 + dep[2]
                    edges.append([stop_times[i][attix['trip_id']]+'^'+stop_times[i][attix['stop_id']],
                                  stop_times[i+1][attix['trip_id']]+'^'+stop_times[i+1][attix['stop_id']],
                                  arrtm-deptm])
                except:
                    if cnt < 10:
                        print('Failed to convert data for row: ', row)
                        cnt += 1
                    else:
                        pass

        print('Finished building line graph: ', time.time() - start1, ' secs ', len(edges), ' edges so far...')
        del stop_times
        start2 = time.time()

        for key in stoptrips.keys():            #for each stop
            strps = stoptrips[key]              #Get the stop trips
            avghw = 24*60/len(strps)
            if avghw < 15:
                lim = 900 
                #lim = 2400
            else:
                lim = 1800

            for ostp, oarr, odep in strps:      # for stop trip, stop arrival, stop departure in each trip of a certain stop. 
                for dstp, darr, ddep in strps:
                    if oarr < ddep and validtrips[ostp] != validtrips[dstp]: #--> no need to create transfer on same route id
                        conntime = max(60, ddep - oarr)
                        if conntime < lim:
                            print(ostp+'^'+key, dstp+'^'+key, conntime+xferpen)
                            edges.append([ostp+'^'+key, dstp+'^'+key, conntime+xferpen])

        print('Finished building valid transfers within stops: ', time.time() - start2, ' secs', len(edges), ' edges so far...')

        with open(dir+'/stops.txt', 'r', encoding='utf-8-sig') as fn:
            reader = csv.reader(fn, delimiter=',')
            fields = next(reader)
            attix = dict(zip(fields, range(len(fields))))
            #print(attix)
            stopdata = {}
            for row in reader:
                stopdata[row[attix['stop_id']]] = [row[attix['stop_name']], float(row[attix['stop_lat']]), float(row[attix['stop_lon']])]

        if filesindir.intersection(['transfers.txt']):            
            #4) load transfer file between stops...
            with open(dir+'/transfers.txt', 'r') as fn:
                reader = csv.reader(fn, delimiter=',')
                fields = next(reader)
                attix = dict(zip(fields, range(len(fields))))
                #print(attix)
                #walkedges = []
                #from_stop_id,to_stop_id,transfer_type
                for row in reader:
                    if row[attix['from_stop_id']] in stoptrips and row[attix['to_stop_id']] in stoptrips:
                        ostoptrips = stoptrips[row[attix['from_stop_id']]]
                        dstoptrips = stoptrips[row[attix['to_stop_id']]]
                        #avghw = 24*60/len(dstoptrips)
                        #if avghw < 15:
                            #lim = 2400
                        #else:
                            #lim = 3600
                        lim = 4800
                        wlktim = computeGCD(stopdata[row[attix['from_stop_id']]][1],stopdata[row[attix['from_stop_id']]][2],stopdata[row[attix['to_stop_id']]][1],stopdata[row[attix['to_stop_id']]][2])*1200*1.25
                        #wlkdis = wlktim/1200
                        for ostp, oarr, odep in ostoptrips:
                            for dstp, darr, ddep in dstoptrips:
                                #trip_id, arr, dep
                                if oarr+wlktim < ddep and validtrips[ostp] != validtrips[dstp]:
                                    conntime = max(60, ddep - oarr)
                                    if conntime < lim:
                                        #validxfers[ostp+key, dstp+key] = conntime
                                        edges.append([ostp+'^'+row[attix['from_stop_id']], dstp+'^'+row[attix['to_stop_id']], conntime+xferpen])
                                        #walkedges.append([ostp+'^'+row[attix['from_stop_id']], dstp+'^'+row[attix['to_stop_id']], wlktim])
        
        print('Finished generating complete search graph in ', time.time()-t0, ' secs')
        G = nx.MultiDiGraph(crs = 'EPSG:32631')
        G.add_weighted_edges_from(edges, 'time')
        del edges
        return G, validservices, validtrips, stoptrips, getstopid, stopdata
    else:
        print('The set of required files for generating the search graph is not complete. Processing aborted!')   #--->if this happens abort..
        return 0

def add_coord_station(G, stopdata):
    for node_id in G.nodes:
        try:
            trip, station = node_id.split('^')
            stop_lat = stopdata[station][1]
            stop_lon = stopdata[station][2]
            G.nodes[node_id]['x'] = stop_lon
            G.nodes[node_id]['y'] = stop_lat
        except:
            print('Failed to find the station id: ', station)

    return G


#dirloc = r"C:\DevResearch\GTFS Builder\gtfs_trimet"
path = os.getcwd()
dirloc = path + '/GTFS/Prova'
saveloc = path + '/GTFS/Graphs/Prova'
xferpen = 650
calmethod = 3
date = '20210326' #'20150611'
day = 'friday'

beg = time.time()
print('Building search graph...')
global G, validservices, validtrips, stoptrips, getstopid, stopdata 
G, validservices, validtrips, stoptrips, getstopid, stopdata = BuildSearchGraph(dirloc, xferpen, calmethod, date, day)
print('Finished building search graph in: ', time.time()-beg, ' secs')

print(G.nodes)
print(G.edges)
print(G.edges[('E1^1', 'E1^2', 0)])
print(G.edges[('E3^1', 'E2^1', 0)])


G = add_coord_station(G,stopdata)

# Dump graph
with open(saveloc + "/" + date + ".p", 'wb') as f:
    pickle.dump(G, f)
