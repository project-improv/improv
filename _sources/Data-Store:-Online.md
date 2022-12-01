## Data Store
 - Class StoreInterface, class Limbo implements PyArrow Plasma data server.
 - Store server is started 'locally' -- need to ensure universal behavior
 - Store client is the dominant portion of the module

### Container
 - Items are uniquely identified by a random object ID
 - Object IDs and their items string names (handles) are referenced in a dict attribute of Limbo
 - Items can be stored as simple objects or as object buffers (PlasmaBuffer)

### Dynamics
 - Simple put/get on single items or small list of items 
 - With buffers, can use memoryview to fill an eg., list, and then seal the buffer when done
 - A get on a buffer will block if the buffer has not yet sealed
 - Listing objects in store (experimental) is possible. Probably create own string listing.

## Limbo data store client

### Data store locations
Each Limbo client maintains a dict of the names of objects it has stored and their plasma ObjectIDs:

`   self.stored contains {object_name:object_id} `

This dict can be obtained by Nexus for coordination among Limbo clients, limbo1.getStored(), and then passed to new clients, limbo2.addStored().

To aid information passing/tracking, each Limbo client also needs to identify the source of this storage:

`   self.stored = {'myself': {name1:id1, name2:id2}, 
                  'acquisition1': {name3:id3, name2:id4}} `

so name domains can overlap between modules. ObjectIDs must continue to be unique. 
