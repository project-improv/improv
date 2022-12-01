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