import psycopg2
import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities

MEMBER_ACTIVITY="""
select tx.*, ty.manager, ty.user_id, ty.entry_date, ty.active from
(select
    ta.*, tb.organization_id as organization_emis, tb.accountable_type as type_emis, tb.balance as balance_emis, tb.accountable_id
from
    (select t0.account_id_emis, t0.account_id_dest, count(*) as n_transf, sum(t0.amount) as amount from
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
        group by account_id_emis, account_id_dest) ta
    left outer join
        (select * from accounts) tb
    on ta.account_id_emis=tb.id) tx
left outer join
    (select * from members ) ty
on tx.accountable_id=ty.id;"""

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
    plt.savefig('global.png')

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

    plt.savefig("local.png")

####################################################

def main():
    ##----Connection to postgres:

    conn = psycopg2.connect(host="localhost",
                            database="timeoverflow",
                            user="postgres",
                            password="test")


    with conn:
        sql1=MEMBER_ACTIVITY
        df = pd.read_sql_query(sql1, conn)  ## Df with all the edges (already grouped by)

        sql2=MEMBER_NODE
        nodeData = pd.read_sql_query(sql2, conn)  ## Df with attributes of nodes

    ## Creating a new attribute of nodes:
    nodeData.manager=nodeData.manager.astype(bool)
    nodeData['special_type'] = nodeData['accountable_type']
    nodeData.loc[nodeData['manager'], 'special_type'] = 'Manager'


    ## Creating the global graph:
    G = nx.from_pandas_edgelist(df, 'account_id_emis', 'account_id_dest', ['amount', 'n_transf'])

    ## Identifying subcommunities (i.e. the banks, in our case different
    ## subcommunities have no connection among them)
    communities=sorted(greedy_modularity_communities(G), key=len, reverse=True)

    ## Count the communities
    print(f"Timeoverflow has {len(communities)} communities.")

    # Set node and edge communities
    set_node_community(G, communities)
    set_edge_community(G)

    ###--- Plotting all the communities: -------------------------
    #plot_global_network(G)


    ## Creating the output dataframe: each line is a community with some variables characterizing it:
    df_out = []

    for i in range(1,len(communities)+1):
        print("We're on community %d" % (i))

        selected_nodes = [x for x,y in G.nodes(data=True) if y['community']==i]
        selected_edges = [(u,v) for u,v,e in G.edges(data=True) if e['community'] ==i]


        H = G.subgraph(selected_nodes)
        if i == 214:
            plot_local_network(nodeData, H)

        #print('n_nodes:', H.number_of_edges())
        print('density: ', nx.density(H))

        most_popular_members = sorted(nx.degree_centrality(H), key = lambda x: (-nx.degree_centrality(H)[x], x))
        df_out.append((i, nx.density(H), H.number_of_nodes(), most_popular_members[:10] ))

    # write the result out as a file
    df_out = pd.DataFrame(df_out, columns=('community', 'density', 'n_nodes','most_popular_members'))
    df_out.to_csv('test.csv', sep='\t', encoding='utf-8')

### TO DO LIST:
# - in the final df find a way to insert the id of the bank (now there is only id of the community)
# - add more variables describing the communities in the final df and also info on the bank itself (demografy, etc.)
# - remove organizations and Managers (?) from most popular members?
# - think about best way to show most popular members (the first n members, or the first n%?  as a vector in one variable or one per column?)
# - improve plots (for example, color of nodes according to special_type, find better visualization for big coomunity)
# - save plots in files with number of the organization
# - Consider if it's better to construct one single graphs and divide it (like now) or construct graphs one by one..
# - ? make directional graphs?
# - ? pre-cleaning  of transactions data?

if __name__=="__main__":
    main()
