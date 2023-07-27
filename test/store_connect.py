import pyarrow.plasma as plasma

# start the store

client = plasma.connect("/tmp/plasma")

object_id = list()

object_id.append(client.put("input 0"))
object_id.append(client.put("input 1"))
object_id.append(client.put("input 2"))
object_id.append(client.put("input 3"))

print(client.get(object_id[3]))
print(client.get(object_id[2]))
print(client.get(object_id[1]))
print(client.get(object_id[0]))

print("\n")

print(client.list())
