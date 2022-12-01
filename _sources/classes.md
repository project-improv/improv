## OnACID

### Attributes
 - params: CNMFParams
 - estimates: Estimates
 - N: int from columns in estimates.A, # of imaged sources
 - M: int from params and self.N, # of total components (background + imaged)
 - dims: (int,int) dimensions of FOV in pixels
 - ind_A: index matrix of Ab stacked matrix; separate from estimates

 - update_counter: array counter for updating shapes, for each component
 - time_neuron_added: array of 'when' component are added
 - time_spend: cumulative time spent running

 - loaded_model: keras model from json; optional

 - Ab_dense_copy: from estimates.Ab_dense. Unused.
 - Ab_copy: from local variable Ab_ (copied from estimates.Ab), then copied back into estimates.Ab after shape updating
 - Ab_epoch: copies of estimates.Ab for each epoch

 - bnd_Y: percentiles of loaded data; used if show movie.
 - bnd_AC: same for A.C
 - bnd_BG: same for b.f. Unused.

 - img_min: min of loaded data
 - img_norm: norm of loaded data, std+median
 
 - t: counter during online fitting; used if show movie and in create_frame method (problematic)
 - t_shapes: list of times for shape updating
 - t_detect: list of times
 - t_motion: list of times
 - comp_upd: list of updated components, used to count updating shapes during fit_next

 - captions: list of captions in show movie. Unused.
 - dview: Unused.

Below from update_num_components: Potentially problematic.
 - rhos_buf: RingBuffer, after update, from estimates.rho_buf. Unused.
 - ind_new_all: Unused.
 - cnn_pos:  Unused.

### Methods
 - fit_next: fits the next frame, updates the object
 - initialize_online: initialize using small portion of the dataset 
 - fit_online: take files and fit in real-time
 - create_frame: currently only used for showing movie, has implicit assumptions
 - _prepare_object: prepares the online object given some estimates
 - save: save object in hdf5 (h5) format

Other methods in online_cnmf:
 - bare_initialization: bypass cnmf to quickly init OnACID (default)
 - seeded_initialization: init OnACID from user specified binary masks
 - HALS4shape: reshape A
 - HALS4activity: get C using block-coordinate descent
 - demix_and_deconvolve: get C using OASIS within b-c descent
 - init_shapes_and_sufficient_stats: estimate shapes on initial batch
 - update_shapes: updates shapes
 - update_num_components: check for new components in residual buffer, adds if needed
 - get_candidate_components: extract new candidate components and test them
 - remove_components_online: remove indexed components 
 - initialize_movie_online: init movie using cnmf 
 - load_OnlineCNMF: load object from save (hdf5)
 - csc_append: appends second csc_matrix to the right of the first one
 - corr: fast correlation
 - rank1nmf: fast rank 1 NMF


## RingBuffer
Implements ring buffer efficiently, inherits from np.ndarray
### Attributes 
 - max_, cur
### Methods
 - append, get_ordered, get_first, get_last_frames


## CNMFParams
Class for setting processing parameters, grouped. Dictionary implementation
### Attributes
 - Primary dicts: data, patch, preprocess, init, spatial, temporal, merging, quality, online, motion
Should all have default values for each specified key.
### Methods 
 - set: add key-value pairs to a given dict (group)
 - get: gets value from group.key
 - get_group: gets full dict of given group
 - change_params: given dict, set all new values
 - to_dict: convert all dicts to single large dict
 - _eq_: define comparison for dicts
(dict_compare method defined in .utilities)