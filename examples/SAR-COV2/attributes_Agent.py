import random

def age_(distr_age):
    """
    distr_age is a dictionary containing age groups with their probabilities
    the function returns a random age according to the probabilities contained in distr_age
    for Barcellona:
    diz_eta={"00-10":8.89,"11-20":8.58,"21-30":13.04,"31-40":15.41,"41-50":15.34,"51-60":13.06,"61-70":10.53,"71-80":8.41,"81-90":5.46,"91-99":1.28}
    """
    distr2=[]
    distr3=[]
    #distr2=[i+sum(distr2) for i in distr]
    for i in distr_age.values():
        distr3.append(i)
        distr2.append(sum(distr3))
    p=random.random()*100  
    for i in range(len(distr2)) :
                if p<distr2[i]:
                    indice_age=i
                    break
                else:                                
                    continue
    A=int(list(distr_age.keys())[indice_age][0]+list(distr_age.keys())[indice_age][1])
    B=int(list(distr_age.keys())[indice_age][-2]+list(distr_age.keys())[indice_age][-1])
    return random.randint(A, B)
    #return list(distr_age.keys())[indice_age]

def job(distr_job):
    """
    distr_job is a dictionary containing workgroups with their probabilities
    the function returns a random workgroup according to the probabilities contained in distr_job
    """
    distr2=[]
    distr3=[]
    #distr2=[i+sum(distr2) for i in distr]
    for i in distr_job.values():
        distr3.append(i)
        distr2.append(sum(distr3))
    p=random.random()*100  
    for i in range(len(distr2)) :
                if p<distr2[i]:
                    indice_job=i
                    break
                else:                                
                    continue
    return list(distr_job.keys())[indice_job]



def create_families(N,distr): 
    """    
    It generates a number of families given a number of individuals from a distribution list
    each term in the distr list represents the probability of generating a family
    with a number of individuals equal to the index of that element of distr
    for Barcellona:
        distr=[19.9 ,23.8 ,20.4, 24.8, 8.9, 2.2]
    """    
    distr2=[]
    distr3=[]
    for i in distr:
        distr3.append(i)
        distr2.append(sum(distr3))
    M=0
    lista_fam=[]
    while M!=N :
        p=random.random()*100
        if N-M<len(distr):
            lista_fam.append(N-M)
        else:   
            for i in range(len(distr2)) :
                if p<distr2[i]:
                    lista_fam.append(i+1)
                    break
                else:                                
                    continue  
        M=sum(lista_fam)
    return lista_fam