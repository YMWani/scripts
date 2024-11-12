import numpy as np
from math import sqrt
import math
import json

class sphere:
    """
    Generate a sphere using different levels of discretization of an icosphere
    Input variables:
    1. sphere radius
    2. discretization level
    """
    def __init__(self, radius, subdivision):
        self.radius = radius
        self.subdivision = subdivision
        self.middle_point_cache = {}

        self.verts = []
        self.faces = []

        self.bondList = []
        self.faceBonds = []
        self.diametric_bonds = []

        self.bondIds = []
        self.r0s = []

        self.generate()

    def vertex(self, x, y, z):
        """ Return vertex coordinates fixed to the unit sphere """
        length = sqrt(x ** 2 + y ** 2 + z ** 2)
        return [(i * self.radius) / length for i in (x, y, z)]

    def icosahedron(self):
        phi = (1. + sqrt(5)) / 2.
        vert = [self.vertex(-1, phi, 0),
                self.vertex(1, phi, 0),
                self.vertex(-1, -phi, 0),
                self.vertex(1, -phi, 0),

                self.vertex(0, -1, phi),
                self.vertex(0, 1, phi),
                self.vertex(0, -1, -phi),
                self.vertex(0, 1, -phi),

                self.vertex(phi, 0, -1),
                self.vertex(phi, 0, 1),
                self.vertex(-phi, 0, -1),
                self.vertex(-phi, 0, 1)]
        faces = [
            # 5 faces around point 0
            [0, 11, 5],
            [0, 5, 1],
            [0, 1, 7],
            [0, 7, 10],
            [0, 10, 11],
            # Adjacent faces
            [1, 5, 9],
            [5, 11, 4],
            [11, 10, 2],
            [10, 7, 6],
            [7, 1, 8],
            # 5 faces around 3
            [3, 9, 4],
            [3, 4, 2],
            [3, 2, 6],
            [3, 6, 8],
            [3, 8, 9],
            # Adjacent faces
            [4, 9, 5],
            [2, 4, 11],
            [6, 2, 10],
            [8, 6, 7],
            [9, 8, 1],
        ]

        self.verts = vert
        self.faces = faces

    def middle_point(self, point_1, point_2):
        """ Find a middle point and project to the unit sphere """

        # We check if we have already cut this edge first to avoid duplicated verts
        smaller_index = min(point_1, point_2)
        greater_index = max(point_1, point_2)

        key = '{0}-{1}'.format(smaller_index, greater_index)

        if key in self.middle_point_cache:
            return self.middle_point_cache[key]

        # If it's not in cache, then we can cut it
        vert_1 = self.verts[point_1]
        vert_2 = self.verts[point_2]
        middle = [sum(i) / 2 for i in zip(vert_1, vert_2)]
        self.verts.append(self.vertex(*middle))

        index = len(self.verts) - 1
        self.middle_point_cache[key] = index

        return index

    def generate_bonds(self, faces, verts):
        # Bonds between the surface monomers
        edges = np.empty((3 * len(faces), 2), dtype=np.int32)
        for i, (a, b, c) in enumerate(faces):
            edges[3 * i + 0] = (a, b) if a <= b else (b, a)
            edges[3 * i + 1] = (a, c) if a <= c else (c, a)
            edges[3 * i + 2] = (b, c) if b <= c else (c, b)
        edges = np.unique(edges, axis=0)
        # create bonds between the center and vertices
        N = self.verts.shape[0]
        centralBonds = np.asarray([[x,N-1] for x in range(N-1)])
        return edges, centralBonds

    def generate(self):
        # Create a regular icosahedron
        self.icosahedron()

        # Subdivision of the faces to create a finer mesh
        face = self.faces
        for i in range(self.subdivision):
            faces_subdiv = []
            for tri in face:
                v1 = self.middle_point(tri[0], tri[1])
                v2 = self.middle_point(tri[1], tri[2])
                v3 = self.middle_point(tri[2], tri[0])

                faces_subdiv.append([tri[0], v1, v3])
                faces_subdiv.append([tri[1], v2, v1])
                faces_subdiv.append([tri[2], v3, v2])
                faces_subdiv.append([v1, v2, v3])
            face = faces_subdiv
        self.faces = face

        # Generate bonds between the vertices
        self.verts = np.asarray(self.verts)
        self.verts = np.append(self.verts, np.array([[0.,0.,0.]]), axis=0)
        self.faceBonds, self.diametric_bonds = self.generate_bonds(self.faces, self.verts)
        self.bondList = np.append(self.faceBonds, self.diametric_bonds, axis=0)

        # Generate bond Ids
        round_to = 3
        dr = np.linalg.norm(self.verts[self.bondList[:,1]] - self.verts[self.bondList[:,0]], axis=1)
        dr = np.round(dr,round_to)
        self.r0s = np.unique(dr)
        self.bondIds = np.zeros(self.bondList.shape[0],
                                dtype=np.int32)
        for i,r0 in enumerate(self.r0s):
            self.bondIds[np.isclose(dr,r0)] = i

def render_geometry(positions, bonds, bondId, bt, r0s, fname):
    import hoomd
    hoomd.context.initialize("--notice-level=0")
    # bt = ['b{}'.format(i) for i in range(len(r0s))]
    snapshot = hoomd.data.make_snapshot(N=len(positions),
                                        box=hoomd.data.boxdim(L=100),
                                        particle_types=['A','B','C'],
                                        bond_types=bt)
    snapshot.particles.position[:] = positions
    snapshot.particles.mass[:] = 5.
    snapshot.particles.typeid[:] = 0
    snapshot.particles.typeid[-1] = 1
    snapshot.particles.diameter[:] = 1.
    snapshot.particles.diameter[-1] = 6.
    snapshot.bonds.resize(len(bonds))
    snapshot.bonds.group[:] = bonds
    snapshot.bonds.typeid[:] = bondId
    hoomd.init.read_snapshot(snapshot)
    all = hoomd.group.all()

    d = hoomd.dump.gsd(fname, period=None, group=all, overwrite=True)
    hoomd.run(0)

if __name__ == "__main__":
    a = 3.0
    subDiv = 2
    sphere = sphere(a, subDiv)

    print(f"For a sphere of radius = {a} and {subDiv} level of discretization, the nearest neighbor distances are: {sphere.r0s} \n")

    #Creating bond dictionary for bond harmonicsbond_lengths
    bond_types = []
    bond_type_lengths = {}
    for i, bond_length in enumerate(sphere.r0s):
        bond_type = 'bond-{}'.format(i)
        bond_types.append(bond_type)
        bond_type_lengths[bond_type] = bond_length
    #Writing and saving json file
    with open("bond_lengths.json", "w") as outfile:
        json.dump(bond_type_lengths, outfile)

    render_geometry(sphere.verts,
                    sphere.bondList,
                    sphere.bondIds,
                    bond_types,
                    sphere.r0s,
                    f"Shape.gsd")
