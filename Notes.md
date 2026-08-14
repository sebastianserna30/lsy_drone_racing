# Current State:

- drone crashes even though there is a conservative radius of  radius*r and the oponnent colission cost is 10k.



# ToDo:

- (J) Dense collision cost (rn is one-hot encoded)
 - (S) cost plots -> cost engineering
- Enforce gate flythrough? We had the case were the drone went in circles around obs1 and then just ignored gate0 and gate1
    - cost progress along track.
    ideas: 
        1. projection along spline
        2. multiple splines. one for each gate2gate segment. Drone should finish one spline before moving to the next.
        3. limit spline on reachable horizon. The drone does not know the full spline, only gets the pieces in runtime.
        
    - (M tip) use the lag/contour costs from MPCC


# ToDo later
- Add downwash to multi_lvl costs -> first make the sphere collision avoidance work and then we try more fancy stuff.s