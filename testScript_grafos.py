# 2020 Col·lectivaT
#
### TO DO LIST:
# - improve plots (for example, color of nodes according to special_type, find
# better visualization for big coomunity)
# - save plots in files with number of the organization
# - ? make directional graphs?
# - ? pre-cleaning  of transactions data?
import os
import psycopg2
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
from datetime import date

### QUERIES: -------------------------------------

MEMBER_ACTIVITY="""
    select 
        t0.account_id_emis, t0.account_id_dest, count(*) as n_transf, sum(t0.amount) as amount 
    from
        (select
                t1.id as id_mov1,  t2.id as id_mov2,
                t1.transfer_id,
                t1.account_id as account_id_emis, t2.account_id as account_id_dest,
                t1.amount, t1.created_at, t1.updated_at
         from
                (select * from movements where amount <0 )t1
         inner join
                (select * from movements where amount >0) t2
         on t1.transfer_id=t2.transfer_id) t0
    group by account_id_emis, account_id_dest;"""

MEMBER_NODE="""
    select
        tx.*,
       (case when tx.accountable_type='Member' then ty.manager else null end) as manager,
       (case when tx.accountable_type='Member' then ty.user_id else null end) as user_id,
       (case when tx.accountable_type='Member' then ty.entry_date else null end) as entry_date,
       (case when tx.accountable_type='Member' then ty.active else null end) as active
    from
       (select ta.*, tb.organization_id, tb.accountable_type, tb.balance, tb.accountable_id from
          (select distinct account_id from movements )ta
        left outer join
          (select * from accounts) tb
        on ta.account_id=tb.id) tx
    left outer join
       (select * from members ) ty
    on tx.accountable_id=ty.id;"""


ORGANIZATION_PROFILE="""
    select
         t1.id, t1.name as bank_name, 
         t2.n_members, t2.PC_known_age, t2.max_age, t2.min_age, t2.avg_age,
         t2.n_females, t2.n_males, t2.n_prefNOansw, t2.n_gender_null, 
         t2.PC_females, t2.PC_males, t2.PC_prefNOansw, t2.PC_gender_null, 
         t2.max_seniority, t2.min_seniority, t2.avg_seniority
    from
         organizations t1
    left outer join
         (select
            organization_id, count(*) as n_records,
            count(distinct id) as n_users, count(distinct member_id) as n_members,
            round(cast(sum(case when age is not null then 1 else 0 end) as numeric)/count(*)*100,1) as PC_known_age,
            max(age) as max_age,
            min(age) as min_age,
            round(avg(age)) as avg_age,
            sum(case when gender ='female' then 1 else 0 end) as n_females,
            sum(case when gender='male' then 1 else 0 end) as n_males,
            sum(case when gender='prefer_not_to_answer' then 1 else 0 end) as n_prefNOansw,
            sum(case when (gender='' or gender is null) then 1 else 0 end) as n_gender_null,
            round(cast(sum(case when gender ='female' then 1 else 0 end) as numeric )/count(*)*100,1) as PC_females,
	    round(cast(sum(case when gender='male' then 1 else 0 end) as numeric )/count(*)*100,1) as PC_males,
	    round(cast(sum(case when gender='prefer_not_to_answer' then 1 else 0 end) as numeric )/count(*)*100,1) as PC_prefNOansw,
	    round(cast(sum(case when (gender='' or gender is null) then 1 else 0 end) as numeric )/count(*)*100,1) as PC_gender_null,
            max(seniority) as max_seniority,
            min(seniority) as min_seniority,  
            round(avg(seniority)) as avg_seniority
         from      
            (select
                ta.id, 
                ta.date_of_birth, case when extract (year from date_of_birth)>1900 then extract ( year from age(ta.date_of_birth)) else null end as age,
                ta.gender, ta.active, ta.created_at, ta.sign_in_count, ta.current_sign_in_at, ta.last_sign_in_at,
                tb.id as member_id, tb.organization_id, tb.manager, tb.entry_date,
                case when extract (year from tb.entry_date)>1900 then extract ( year from age(tb.entry_date)) else null end as seniority,
                tb.member_uid, tb.active as active_member
             from
                users ta
             left outer join
                members tb
             on ta.id=tb.user_id) tt
         group by organization_id) t2
    on t1.id=t2.organization_id;"""



### FUNCTIONS: -------------------------------------

def set_node_community(G, communities):
        '''Add community to node attributes'''
        for c, v_c in enumerate(communities):
            for v in v_c:
                # Add 1 to save 0 for external edges
                G.nodes[v]['community'] = c + 1

def set_edge_community(G):
        '''Find internal edges and add their community to their attributes'''
        for v, w, in G.edges:
            if G.nodes[v]['community'] == G.nodes[w]['community']:
                # Internal edge, mark with community
                G.edges[v, w]['community'] = G.nodes[v]['community']
            else:
                # External edge, mark as 0
                G.edges[v, w]['community'] = 0

def get_color(i, r_off=1, g_off=1, b_off=1):
        '''Assign a color to a vertex.'''
        r0, g0, b0 = 0, 0, 0
        n = 16
        low, high = 0.1, 0.9
        span = high - low
        r = low + span * (((i + r_off) * 3) % n) / (n - 1)
        g = low + span * (((i + g_off) * 5) % n) / (n - 1)
        b = low + span * (((i + b_off) * 7) % n) / (n - 1)
        return (r, g, b)

def plot_global_network(G):
    # Set community color for edges between members of the same community
    # (internal) and intra-community edges (external)
    external = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] == 0]
    internal = [(v, w) for v, w in G.edges if G.edges[v, w]['community'] > 0]
    internal_color = ['black' for e in internal]
    node_color = [get_color(G.nodes[v]['community']) for v in G.nodes]

    G_pos = nx.spring_layout(G)

    plt.rcParams.update({'figure.figsize': (20, 20)})
    # Draw external edges
    nx.draw_networkx(
            G,
            pos=G_pos,
            node_size=0,
            edgelist=external,
            edge_color="silver")
    # Draw nodes and internal edges
    nx.draw_networkx(
            G,
            pos=G_pos,
            node_color=node_color,
            edgelist=internal,
            edge_color=internal_color)
    plt.savefig('results/global.png')

def plot_local_network(nodeData, H):
    ###------plot of the community : --------------------------

    plt.rcParams['figure.figsize'] = [10, 10]
                  
    edge_widthA = [0.0001 * H[u][v]['amount'] for u, v in H.edges()] 
    edge_widthN = [0.5 * H[u][v]['n_transf'] for u, v in H.edges()] 
    
    nx.set_node_attributes(H, nodeData.set_index('account_id').to_dict('index'))

    node_color = [H.degree(v) for v in H]

    pos = nx.spring_layout(H, scale=2)

    nx.draw(H, pos, width=edge_widthN) ## node_color=node_color)
    node_labels = nx.get_node_attributes(H,'special_type')
    nx.draw_networkx_labels(H,pos, node_labels)
    edge_labelsA = nx.get_edge_attributes(H,'n_transf')
    nx.draw_networkx_edge_labels(H,pos,edge_labelsA)  

    plt.savefig("results/local.png")

####################################################

def main(psql_config):
    ##----Connection to postgres:
    server, to_user, to_password, to_database = psql_config
    conn = psycopg2.connect(host=server,
                            database=to_database,
                            user=to_user,
                            password=to_password)


    with conn:
        sql1=MEMBER_ACTIVITY
        df = pd.read_sql_query(sql1, conn)  ## df with all the edges (already grouped by)

        sql2=MEMBER_NODE
        nodeData = pd.read_sql_query(sql2, conn)  ## df with attributes of nodes

        sql3=ORGANIZATION_PROFILE
        df_banks= pd.read_sql_query(sql3, conn)  ## df with list of all banks and some variables characterizing them
        

    ## Creating a new attribute of nodes:
    nodeData.manager=nodeData.manager.astype(bool)
    nodeData['special_type'] = nodeData['accountable_type']
    nodeData.loc[nodeData['manager'], 'special_type'] = 'Manager'

    ## Saving df with nodes and their attributes
    nodeData.to_csv('results/nodes.csv', sep='\t', encoding='utf-8')

    ## Adding to the edges df the id of the bank 
    dff= pd.merge(df, nodeData[['account_id', 'organization_id']], left_on='account_id_emis', right_on='account_id')
    dff= pd.merge(dff, nodeData[['account_id', 'organization_id']], left_on='account_id_dest', right_on='account_id')

    dff['bank_id']=dff['organization_id_x']
    dff.loc[dff['organization_id_x'].isna() & dff['organization_id_y'].notna(), 'bank_id'] = dff.organization_id_y

    df=dff.drop(columns=['account_id_x', 'account_id_y', 'organization_id_x', 'organization_id_y'])

    df=df[df['bank_id'].notna()]
    #print('After cleaning, the bank ids are: ', df.bank_id.unique())

    ## Saving df with edges and their attributes
    df.to_csv('results/edges.csv', sep='\t', encoding='utf-8')
    
    ## Count the communities with at least one transfer in the whole history of TO:
    print(f"Timeoverflow has {len(df.bank_id.unique())} communities.")
    
    ## Creating the networks dataframe: each line is a bank with some variables characterizing its network:
    df_redes=[]

    column_names = ['organization_id', 'account_id', 'special_type','centrality']
    df_cc = pd.DataFrame(columns = column_names)


    for i in df.bank_id.unique():
            
            dd=df[df['bank_id']==i] #df with edges between members of that specific bank
            n_transf=sum(dd.n_transf)
            
            print("Analizing organization %d   --------------------------------" % (i))
           
            
            G1 = nx.from_pandas_edgelist(dd, 'account_id_emis', 'account_id_dest', ['amount', 'n_transf', 'bank_id'])
     
            nx.set_node_attributes(G1, nodeData.set_index('account_id').to_dict('index'))
            

            print("Its network has %d edges and %d nodes" % (G1.number_of_edges(),G1.number_of_nodes()) )
            print('and a density of: ', nx.density(G1))

           

            ## Create ranking of members according to their centrality inside the bank:
            #most_active_members = sorted(nx.degree_centrality(G1), key = lambda x: (-nx.degree_centrality(G1)[x], x)) ##list of ids in descending order of centrality
            dfi=[]
            dfi = pd.DataFrame(nx.degree_centrality(G1).items(), columns=['node_id', 'centrality'])                       
            dfi = pd.merge(dfi, nodeData[['account_id', 'special_type', 'organization_id']], left_on='node_id', right_on='account_id')

            dfi.drop(columns=['node_id'])
            dfi = dfi[['organization_id', 'account_id', 'special_type','centrality']]
            dfi = dfi.sort_values(by=['centrality'],ascending=False)   
            dfi = dfi[dfi.organization_id.notna()]
            
            df_cc = pd.concat([df_cc, dfi])
            
            if i == 214:
                   plot_local_network(nodeData, G1)
            
            ## Count members with degree centrality >20%
            count_c20 = sum(map(lambda x : x>0.2, list(nx.degree_centrality(G1).values())))


            ## Array with the centrality values of all the nodes
            cc=np.array(list(nx.degree_centrality(G1).values()))
            
            df_redes.append((i, nx.density(G1), n_transf, G1.number_of_edges(), G1.number_of_nodes(), 
                           cc.mean(), cc.min(), cc.max(), count_c20, count_c20/len(cc)*100))
        
            

    ## Set column names of the network df
    df_redes = pd.DataFrame(df_redes, columns=('bank_id', 'density','n_transf', 'n_edges', 'n_nodes','avg_centrality', 'min_centrality', 'max_centrality', 'n_popular_members', 'PC_popular_members'))

    df_cc.organization_id=df_cc.organization_id.astype(int)    
    df_cc.to_csv('results/members_centralities.csv', sep='\t', encoding='utf-8')
       
    ## Join the network df with the list of banks and their general characteristics and write the result out as a file
    df_banks=df_banks.merge(df_redes, left_on='id', right_on='bank_id', how='left')
    df_banks['date']=date.today()
    df_banks.to_csv('results/organizations_profiles.csv', sep='\t', encoding='utf-8')
   

    
if __name__=="__main__":
    psql_config = (os.environ.get('TO_DB_SERVER'),
                   os.environ.get('TO_DB_USER'),
                   os.environ.get('TO_DB_PASSWORD'),
                   os.environ.get('TO_DB_NAME'))
    for element in psql_config:
        if not element:
            raise ValueError('TO_DB_SERVER, TO_DB_USER, TO_DB_PASSWORD and TO_DB_NAME '\
                             'has to be set as environment variables.')
    main(psql_config)
