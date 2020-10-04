Multi-relational graphs are a more general and prevalent form of graphs where each edge has a label and direction associated with it. However, the primary focus has been on handling simple undirected graphs and the approaches tackling multi relational hetero-graphs often suffer from over-parameterization (like [R-GCN](https://arxiv.org/pdf/1703.06103.pdf)) with most of them restricted to learning node representations only. Hence, such methods are not directly applicable for tasks such as link prediction which require relation embedding vectors. By definition, it's clear that relations are extremely important in these hetero-graphs (eg. KGs). The goal is to learn representations in a relational graph of both the nodes and relations by jointly embedding them.

CompGCN introduced in this paper is a method leveraging a variety of entity-relation composition operations from Knowledge Graph Embedding techniques (TransE, TransR, RotatE, DistMul etc.) and scales with the number of relations. 

<p align="center">
  <img width="70%" src="https://github.com/malllabiisc/CompGCN/raw/master/overview.png" />
</p>


<center><a href="https://arxiv.org/abs/1911.03082"> Composition-Based Multi-Relational Graph Convolutional Networks (CompGCN) </a></center>


## KG Embeddings (KGEs)

A knowledge graph (KG) is a directed heterogeneous multigraph whose node and relation types have domain-specific semantics. Knowledge graph embeddings (KGEs) are low-dimensional representations of the entities and relations in a knowledge graph. They provide a generalizable context about the overall KG that can be used to infer relations. Most of KG embedding approaches define a score function and train node and relation embeddings such that valid triples are assigned a higher score than the invalid ones.

<p align="center">
  <img width="25%" src="https://miro.medium.com/max/231/1*U97Dzy42ZBLbWOX40rNedQ.png" />
</p>

Above figure illustrates an example of TransE method for KG Embedding. TransE is based on Additive score function. There are other methods which are based on Multiplicative or Neural scoring.

### GCN on Multi-Relational Graphs

For a multi-relational graph G = (V,R,E,X), where R denotes the set of relations, and each edge (u, v, r) represents that the relation r ∈ R exist from node u to v. The GCN formulation as devised by Marcheggiani & Titov (2017) is based on the assumption that information in a directed edge flows along both directions. Hence, for each edge (u, v, r) ∈ E, an inverse edge (v, u, r−1) is included in G. The representations obtained after k layers of directed GCN is given by

<a href="https://www.codecogs.com/eqnedit.php?latex=H^{k&plus;1}&space;=&space;\sigma(\hat{A}H^{k}{W^{i}_r})" target="_blank"><img src="https://latex.codecogs.com/png.latex?H^{k&plus;1}&space;=&space;\sigma(\hat{A}H^{k}{W^{i}_r})" title="H^{k+1} = \sigma(\hat{A}H^{k}{W^{i}_r})" /></a>

Here, <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;{W^{k}_r}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;{W^{k}_r}" title="{W^{k}_r}" /></a> denotes the relation specific parameters of the model. However, the above formulation leads to over-parameterization with an increase in the number of relations and hence, Marcheggiani & Titov (2017) use direction-specific weight matrices. Schlichtkrull et al. (2017) in [R-GCN](https://arxiv.org/pdf/1703.06103.pdf) address over-parameterization by proposing basis and block-diagonal decomposition of Wrk.


<p align="center">
  <img width="40%" src="https://www.cellstrat.com/wp-content/uploads/2020/06/Single-Node.png" />
</p>

Activations (d-dimensional vectors) from neighboring nodes (dark blue) are gathered and then transformed for each relation type individually (for both in- and out- edges). The resulting representation (green) is accumulated in a (normalized) sum and passed through an activation function (such as the ReLU).


### CompGCN Formulation

Following Directed-GCN and Relational-GCN, CompGCN also allows the information in a directed edge to flow along both directions. Hence, E and R are extended with corresponding inverse edges and relations, i.e., <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\left.\mathcal{E}^{\prime}=\mathcal{E}&space;\cup\left\{\left(v,&space;u,&space;r^{-1}\right)&space;\mid(u,&space;v,&space;r)&space;\in&space;\mathcal{E}\right\}&space;\cup\{(u,&space;u,&space;\top)&space;\mid&space;u&space;\in&space;\mathcal{V})\right\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\left.\mathcal{E}^{\prime}=\mathcal{E}&space;\cup\left\{\left(v,&space;u,&space;r^{-1}\right)&space;\mid(u,&space;v,&space;r)&space;\in&space;\mathcal{E}\right\}&space;\cup\{(u,&space;u,&space;\top)&space;\mid&space;u&space;\in&space;\mathcal{V})\right\}" title="\left.\mathcal{E}^{\prime}=\mathcal{E} \cup\left\{\left(v, u, r^{-1}\right) \mid(u, v, r) \in \mathcal{E}\right\} \cup\{(u, u, \top) \mid u \in \mathcal{V})\right\}" /></a> and <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{R}^{\prime}=\mathcal{R}&space;\cup&space;\mathcal{R}_{\text&space;{inv}}&space;\cup\{\top\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{R}^{\prime}=\mathcal{R}&space;\cup&space;\mathcal{R}_{\text&space;{inv}}&space;\cup\{\top\}" title="\mathcal{R}^{\prime}=\mathcal{R} \cup \mathcal{R}_{\text {inv}} \cup\{\top\}" /></a> where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;\mathcal{R}_{\text&space;{inv}}=\left\{r^{-1}&space;\mid&space;r&space;\in&space;\mathcal{R}\right\}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;\mathcal{R}_{\text&space;{inv}}=\left\{r^{-1}&space;\mid&space;r&space;\in&space;\mathcal{R}\right\}" title="\mathcal{R}_{\text {inv}}=\left\{r^{-1} \mid r \in \mathcal{R}\right\}" /></a> denotes the inverse relations and T indicates the self loop.

Unlike most of the existing methods which embed only nodes in the graph, COMPGCN learns a d-dimensional representation **hr** along with node embeddings **hv**. Representing relations as vectors alleviates the problem of over-parameterization while applying GCNs on relational graphs. Further, it allows COMPGCN to exploit any available relation features (Z) as initial representations.

To incorporate relational embeddings into GCN, the composition operators from KG Embeddings are used.


<center><a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{h}_{v}=f\left(\sum_{(u,&space;r)&space;\in&space;\mathcal{N}(v)}&space;\boldsymbol{W}_{\lambda(r)}&space;\phi\left(\boldsymbol{x}_{u},&space;\boldsymbol{z}_{r}\right)\right)" target="_blank"><img src="https://latex.codecogs.com/png.latex?\boldsymbol{h}_{v}=f\left(\sum_{(u,&space;r)&space;\in&space;\mathcal{N}(v)}&space;\boldsymbol{W}_{\lambda(r)}&space;\phi\left(\boldsymbol{x}_{u},&space;\boldsymbol{z}_{r}\right)\right)" title="\boldsymbol{h}_{v}=f\left(\sum_{(u, r) \in \mathcal{N}(v)} \boldsymbol{W}_{\lambda(r)} \phi\left(\boldsymbol{x}_{u}, \boldsymbol{z}_{r}\right)\right)" /></a></center>

where <a href="https://www.codecogs.com/eqnedit.php?latex=\inline&space;x_u,&space;z_r" target="_blank"><img src="https://latex.codecogs.com/png.latex?\inline&space;x_u,&space;z_r" title="x_u, z_r" /></a> denotes initial features for node u and relation r respectively, hv denotes the updated representation of node v, and Wλ(r) is a relation-type specific parameter. In COMPGCN, we use direction specific weights, i.e., λ(r) = dir(r), given as:

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{W}_{\mathrm{dir}(r)}=\left\{\begin{array}{ll}&space;\boldsymbol{W}_{O},&space;&&space;r&space;\in&space;\mathcal{R}&space;\\&space;\boldsymbol{W}_{I},&space;&&space;r&space;\in&space;\mathcal{R}_{\text&space;{inv}}&space;\\&space;\boldsymbol{W}_{S},&space;&&space;r=\mathrm{T}(\text&space;{self}&space;\text&space;{-loop})&space;\end{array}\right." target="_blank"><img src="https://latex.codecogs.com/png.latex?\boldsymbol{W}_{\mathrm{dir}(r)}=\left\{\begin{array}{ll}&space;\boldsymbol{W}_{O},&space;&&space;r&space;\in&space;\mathcal{R}&space;\\&space;\boldsymbol{W}_{I},&space;&&space;r&space;\in&space;\mathcal{R}_{\text&space;{inv}}&space;\\&space;\boldsymbol{W}_{S},&space;&&space;r=\mathrm{T}(\text&space;{self}&space;\text&space;{-loop})&space;\end{array}\right." title="\boldsymbol{W}_{\mathrm{dir}(r)}=\left\{\begin{array}{ll} \boldsymbol{W}_{O}, & r \in \mathcal{R} \\ \boldsymbol{W}_{I}, & r \in \mathcal{R}_{\text {inv}} \\ \boldsymbol{W}_{S}, & r=\mathrm{T}(\text {self} \text {-loop}) \end{array}\right." /></a></center>


Further, in CompGCN, after the node embedding update, the relation embeddings are also transformed as follows:

<center><a href="https://www.codecogs.com/eqnedit.php?latex=\boldsymbol{h}_{r}=\boldsymbol{W}_{\text&space;{rel&space;}}&space;\boldsymbol{z}_{r}" target="_blank"><img src="https://latex.codecogs.com/png.latex?\boldsymbol{h}_{r}=\boldsymbol{W}_{\text&space;{rel&space;}}&space;\boldsymbol{z}_{r}" title="\boldsymbol{h}_{r}=\boldsymbol{W}_{\text {rel }} \boldsymbol{z}_{r}" /></a></center>

where Wrel is a learnable transformation matrix which projects all the relations to the same embedding space as nodes and allows them to be utilized in the next CompGCN layer.


## Key Contributions
- A novel Graph Convolutional based framework for multi-relational graphs which leverages a variety of composition operators from Knowledge Graph embedding techniques to jointly embed nodes and relations in a graph.

- Generalizes several existing multi-relational GCN methods.

- Alleviates the problem of over-parameterization by sharing relation embeddings across layers and using basis decomposition.

- The choice of composition operation is important in deciding the quality of the learned embeddings. Hence, superior composition operations for Knowledge Graphs developed in future can be adopted to improve COMPGCN’s performance further.


## References

- [CompGCN](https://arxiv.org/abs/1911.03082)

- [Intro to KG Embedding](https://towardsdatascience.com/introduction-to-knowledge-graph-embedding-with-dgl-ke-77ace6fb60ef)

- [Knowledge-Base-Relation-Prediction](https://deepakn97.github.io/blog/2019/Knowledge-Base-Relation-Prediction/)

- [Overview Relational GCN](https://www.cellstrat.com/2020/06/23/relational-gcn/)

- [Graph Nets: Blog Series](https://github.com/dsgiitr/graph_nets)
