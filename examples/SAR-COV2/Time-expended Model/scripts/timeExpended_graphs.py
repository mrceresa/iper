import csv
import time, os
import ast
from math import *
import networkx as nx
import matplotlib as plt
import pickle 
import osmnx as ox
start = time.time()

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
       
        #4)Build linear edges: From departure station = x trip = y to arrival station = x+1 trip = y
        start1 = time.time()
        cnt = 0
        for i in range(0, len(stop_times)-1):
            if stop_times[i][attix['trip_id']] == stop_times[i+1][attix['trip_id']]: #Check stop sequence is truly concadenated. 
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

        #5) Build valid transfers edges within stops
        for key in stoptrips.keys():            #for each stop
            strps = stoptrips[key]              #Get the stop trips
            lim = 1800
            for ostp, oarr, odep in strps:      # for stop trip, stop arrival, stop departure in each trip of a certain stop. 
                for dstp, darr, ddep in strps:
                    if oarr < ddep and validtrips[ostp] != validtrips[dstp]: #--> no need to create transfer on same route id
                        conntime = max(60, ddep - oarr)
                        if conntime < lim:
                            print(ostp+'^'+key, dstp+'^'+key, conntime+xferpen)
                            edges.append([ostp+'^'+key, dstp+'^'+key, conntime+xferpen])

        print('Finished building valid transfers within stops: ', time.time() - start2, ' secs', len(edges), ' edges so far...')

        # Get the data for each stop 
        with open(dir+'/stops.txt', 'r', encoding='utf-8-sig') as fn:
            reader = csv.reader(fn, delimiter=',')
            fields = next(reader)
            attix = dict(zip(fields, range(len(fields))))
            #print(attix)
            stopdata = {}
            for row in reader:
                stopdata[row[attix['stop_id']]] = [row[attix['stop_name']], float(row[attix['stop_lat']]), float(row[attix['stop_lon']])]

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


path = os.getcwd()
dirlocBaixLlobregat = path + '/GTFS/TBX'         #Dir where the GTFS txt files are.
dirlocBesos = path + '/GTFS/TBS'
dirlocprova = path + '/GTFS/Prova'
saveloc = path + '/GTFS/Graphs/Prova'
xferpen = 650
calmethod = 3
date = '20210326'
day = 'friday'
save = False

beg = time.time()
print('Building search graph...')
G, validservices, validtrips, stoptrips, getstopid, stopdata = BuildSearchGraph(dirlocprova, xferpen, calmethod, date, day)
G = add_coord_station(G,stopdata)
print('Finished building search graph in: ', time.time()-beg, ' secs')

print(G.nodes)
print(G.edges)
try:
    #Print one linear edge and one transfer edge for the prova graph
    print(G.edges[('E1^1', 'E1^2', 0)])
    print(G.edges[('E3^1', 'E2^1', 0)])
except:
    #print one edge for any other graph.
    print(G.edges[(list(G.nodes)[0], list(G.nodes)[1], 0)])

ox.plot_graph(G)

if save == True:
    # Dump graph
    with open(saveloc + "/" + date + ".p", 'wb') as f:
        pickle.dump(G, f)
