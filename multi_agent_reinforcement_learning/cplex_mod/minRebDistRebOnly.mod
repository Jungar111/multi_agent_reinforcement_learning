execute
{
  cplex.threads=4; // Number of cores on your CPU limits this.
}
tuple Edge{
  int i;
  int j;
}

tuple edgeAttrTuple{
    int i;
    int j;
    int t;
}

tuple accTuple{
  int i;
  float n;
}

string path = ...;

{edgeAttrTuple} edgeAttr = ...;
{accTuple} acc_init_tuple = ...;
{accTuple} acc_actor_tuple = ...;

{Edge} edge = {<i,j>|<i,j,t> in edgeAttr};
{int} region = {i|<i,v> in acc_init_tuple};

float time[edge] = [<i,j>:t|<i,j,t> in edgeAttr]; // TODO: distance --> we have no distance (replace with time?)
float desiredVehicles[region] = [i:v|<i,v> in acc_actor_tuple]; // TODO: desiredVehicles
//float accInit[region] = [i:v|<i,v> in accInitTuple];
float vehicles[region] = [i:v|<i,v> in acc_init_tuple]; // TODO: vehicles

dvar int+ demandFlow[edge];
dvar int+ rebFlow[edge];

minimize(sum(e in edge) (rebFlow[e]*time[e]));
subject to
{
  forall(i in region)
    {
    sum(e in edge: e.i==i && e.i!=e.j) (rebFlow[<e.j, e.i>] - rebFlow[<e.i, e.j>]) >= desiredVehicles[i] - vehicles[i];
    sum(e in edge: e.i==i && e.i!=e.j) rebFlow[<e.i, e.j>] <= vehicles[i];
    }
}

main {
  thisOplModel.generate();
  cplex.solve();
  var ofile = new IloOplOutputFile(thisOplModel.path);
  ofile.write("flow=[")
  for(var e in thisOplModel.edge)
       {
         ofile.write("(");
         ofile.write(e.i);
         ofile.write(",");
         ofile.write(e.j);
         ofile.write(",");
         ofile.write(thisOplModel.rebFlow[e]);
         ofile.write(")");
       }
  ofile.writeln("];")
  ofile.close();
}
