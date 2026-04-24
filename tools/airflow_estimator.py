import warnings
import adios4dolfinx
import numpy as np
import pandas as pd

from mpi4py import MPI
from pathlib import Path
from scipy.spatial import cKDTree
from basix.ufl import element, mixed_element
from dolfinx import fem, mesh
from airflow_solvers import (
    AirflowSolverConfig,
    LinearLeastSquaresSolver,
    MinimumResidualSolver,
    WeakPenaltySolver,
)

class AirflowMeasurements:

    def __init__(self, estimator: "AirflowEstimator"):
        self.estimator = estimator

    def set_from_csv(self,
                     samples_csv: str | Path,
                     count: int | None = None,
                     noise_std: float | None = None,
                     max_xy_dist: float | None = None) -> dict[str, float]:
        samples_csv = Path(samples_csv).resolve(strict=True)
        df = pd.read_csv(samples_csv)
        if count is not None:
            count = int(count)
            if count < 1:
                raise ValueError(f"count must be at least 1, got {count}.")
            if count > len(df):
                raise ValueError(
                    f"Requested {count} measurements from {samples_csv.name}, but file only contains {len(df)} rows."
                )
            df = df.iloc[:count].copy()

        required = ["x", "y", "wind_x", "wind_y"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in {samples_csv.name}: {missing}. "
                f"Expected at least {required}."
            )
        if len(df) == 0:
            raise ValueError(f"No sample rows found in {samples_csv}")

        samples_xy = df[["x", "y"]].to_numpy(dtype=float)
        samples_uv = df[["wind_x", "wind_y"]].to_numpy(dtype=float, copy=True)
        
        if noise_std is not None: 
            samples_uv += np.random.normal(0, noise_std, (len(df), 2))

        node_xy = np.asarray(self.estimator.V.tabulate_dof_coordinates(), dtype=float)[:, :2]
        tree = cKDTree(node_xy)
        dist, node_ids = tree.query(samples_xy, k=1, p=2.0, workers=-1)

        max_dist = float(np.max(dist))
        if max_xy_dist is not None and max_dist > max_xy_dist:
            raise ValueError(
                f"Maximum XY mapping distance exceeded: {max_dist:.6g} > {max_xy_dist:.6g}"
            )
        n_input = int(len(node_ids))
        _, first_idx = np.unique(node_ids, return_index=True)
        keep = np.sort(first_idx)
        n_dropped = n_input - int(len(keep))
        node_ids = node_ids[keep]
        samples_uv = samples_uv[keep]
        dist = dist[keep]

        x_ids = node_ids * 2
        y_ids = node_ids * 2 + 1
        velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten().astype(np.int32)
        measurement_ids_W = self.estimator.V_to_W[velocity_ids_V]
        measurement_values = np.stack((samples_uv[:, 0], samples_uv[:, 1]), axis=1).flatten()

        self.estimator.set_measurements(
            measurement_ids_W=measurement_ids_W,
            measurement_values=measurement_values,
            clear_existing=True,
        )

        return {
            "n_input_samples": float(n_input),
            "n_used_samples": float(len(node_ids)),
            "n_dropped_duplicate_nodes": float(n_dropped),
            "max_xy_dist": float(np.max(dist)) if len(dist) else 0.0,
        }

    def reset_random(self, p: int, seed: int | None = None):
        if self.estimator.ground_truth is None:
            raise ValueError("No ground truth set. Use set_ground_truth() first.")

        rng = np.random.default_rng(seed)
        coords_V = self.estimator.V.tabulate_dof_coordinates()
        sample_ids = rng.choice(len(coords_V), size=p, replace=False)

        x_ids = sample_ids * 2
        y_ids = sample_ids * 2 + 1
        velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten()
        measurement_ids_W = self.estimator.V_to_W[velocity_ids_V]
        measurement_values = self.estimator.ground_truth.x.array[measurement_ids_W]

        self.estimator.set_measurements(
            measurement_ids_W=measurement_ids_W,
            measurement_values=measurement_values,
            clear_existing=True,
        )

    def coordinates(self) -> np.ndarray:
        coords_P2 = self.estimator.V.tabulate_dof_coordinates()
        W_to_V = {w: v for v, w in enumerate(self.estimator.V_to_W)}
        measured_v_ids = [W_to_V[i] for i in self.estimator.measurement_ids_W if i in W_to_V]
        measured_v_ids_unique = np.unique(np.array(measured_v_ids) // self.estimator.domain.geometry.dim)
        return coords_P2[measured_v_ids_unique]


class AirflowEstimator:

    def __init__(self,
                 domain: mesh.Mesh,
                 facet_tags: mesh.MeshTags,
                 w_measured: fem.Function,
                 measurement_ids_W: list):
        """
        Initialisiert den Estimator OHNE Boundary Conditions.
        Alle Funktionsräume werden aufgebaut, damit man sie direkt für BCs nutzen kann.
        """
        self.domain: mesh.Mesh = domain
        self.measurement_ids_W = np.asarray(measurement_ids_W, dtype=np.int32)
        self.w_measured = w_measured

        (self.W, self.W0, self.W1,
         self.V, self.Q,
         self.V_to_W, self.Q_to_W) = self.build_mixed_space(domain)
        
        self.domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
        
        self.viscosity = 1e-5
        self.weight_misfit = 1e2
        self.weight_pde_res = 1e0
        self.weight_reg = 1e-2
        self.weight_boundary = 1e4
        self.regularization_mode = "smooth"

        self.bcs: list[fem.DirichletBC] = []
        self.facet_tags = facet_tags
        self.w_final: fem.Function | None = None
        self.ground_truth: fem.Function | None = None
        self._boundary_name_to_id: dict[str, int] = {}
        self.measurements = AirflowMeasurements(self)

    @classmethod
    def from_domain(cls,
                    domain: mesh.Mesh,
                    facet_tags: mesh.MeshTags,
                    meshfile: str | Path | None = None,
                    ground_truth: fem.Function | None = None):
        """
        Construct estimator from fem domain data from .msh file.

        Parameters
        ----------
        domain : mesh.Mesh
            Existing simulation mesh.
        facet_tags : mesh.MeshTags
            Optional facet meshtags matching the domain.
        meshfile : str | Path | None
            Optional Gmsh .msh file to recover physical group names.
        ground_truth : fem.Function | None
            Optional reference field in either V or W space.
        """
        W, _, _, V, _, _, _ = cls.build_mixed_space(domain)
        w_measured = fem.Function(W)
        w_measured.x.array[:] = 0.0

        est = cls(
            domain,
            facet_tags,
            w_measured,
            np.array([], dtype=np.int32)
        )

        if meshfile is not None:
            est._boundary_name_to_id = cls._read_physical_name_map(meshfile)

        if ground_truth is not None:
            if ground_truth.function_space == V:
                w_truth = fem.Function(W)
                w_truth.x.array[:] = 0.0
                w_truth.sub(0).interpolate(ground_truth)
                est.set_ground_truth(w_truth)
            else:
                est.set_ground_truth(ground_truth)

        return est

    @classmethod
    def from_bp(cls,
                bp_path: Path,
                p: int | None = 0,
                seed: int | None = 0,
                meshtags_name: str = "facet_tags",
                fun_name: str | None = "velocity",
                meshfile: Path | None = None):
        """
        Construct estimator from ADIOS bp file.

        Parameters
        ----------
        bp_path : Path
            ADIOS .bp file containing mesh, velocity, facet tags.
        p : int
            Number of velocity measurement points.
        seed : int
            Random seed for measurement sampling.
        fun_name : str
            Name of the velocity function in the file.
        meshtags_name : str
            Name of facet tags in the file.
        meshfile : Path | None
            Optional Gmsh .msh file to recover physical group names
            (e.g. 'Walls', 'Outflow'). If provided, we build a
            name -> id mapping used by set_*_bc helpers.
        """
        domain = adios4dolfinx.read_mesh(bp_path, MPI.COMM_WORLD)
        W, _, _, V, _, V_to_W, _ = cls.build_mixed_space(domain)

        u_true = fem.Function(V)
        adios4dolfinx.read_function(bp_path, u_true, name=fun_name)

        w_true = fem.Function(W)
        w_true.sub(0).interpolate(u_true)

        coords_P2 = V.tabulate_dof_coordinates()
        rng = np.random.default_rng(seed)
        sample_ids = rng.choice(len(coords_P2), size=p, replace=False)
        x_ids, y_ids = sample_ids * 2, sample_ids * 2 + 1
        velocity_ids_V = np.stack((x_ids, y_ids)).T.flatten()
        measurement_ids_W = np.asarray(V_to_W, dtype=np.int32)[velocity_ids_V]

        w_measured = fem.Function(W)
        w_measured.x.array[:] = 0.0
        w_measured.x.array[measurement_ids_W] = w_true.x.array[measurement_ids_W]

        try:
            tags = adios4dolfinx.read_meshtags(bp_path, domain, meshtags_name)
        except:
            raise ValueError(f"No meshtags found under name '{meshtags_name}'")
    
        est = cls(domain, tags, w_measured, measurement_ids_W)
        est.set_ground_truth(w_true)

        if meshfile is not None:
            est._boundary_name_to_id = cls._read_physical_name_map(meshfile)
        else:
            est._boundary_name_to_id = {}
        return est
        
    def _ensure_boundary_name_map(self):
        if self.facet_tags is None:
            raise RuntimeError("facet_tags is not set.")
        if not self._boundary_name_to_id:
            raise RuntimeError(
                "No boundary name->id mapping available. "
                "Provide meshfile during construction or set _boundary_name_to_id manually."
            )

    def set_no_slip_bc(self, wall_names: str | list[str]):
        """
        Apply no-slip (u=0) boundary condition on the given physical boundaries.

        Parameters
        ----------
        wall_names : str or list[str]
            Physical group names, e.g. 'Walls' or ['Walls', 'Obstacles'].
        """
        self._ensure_boundary_name_map()

        if isinstance(wall_names, str):
            names = [wall_names]
        else:
            names = list(wall_names)

        facets = np.concatenate([
            self.facet_tags.find(self._boundary_name_to_id[name]) for name in names
        ])

        u_D = fem.Function(self.V)
        u_D.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W0, self.V),
                                           self.domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(u_D, dofs, self.W0)
        self.add_dirichlet_bc(bc)
        return bc

    def set_zero_pressure_bc(self, outlet_names: str | list[str]):
        """
        Apply p=0 boundary condition on the given outlet boundaries.

        Parameters
        ----------
        outlet_names : str or list[str]
            Physical group names, e.g. 'Outflow'.
        """
        self._ensure_boundary_name_map()

        if isinstance(outlet_names, str):
            names = [outlet_names]
        else:
            names = list(outlet_names)

        facets = np.concatenate([
            self.facet_tags.find(self._boundary_name_to_id[name]) for name in names
        ])

        p_zero = fem.Function(self.Q)
        p_zero.x.array[:] = 0.0

        dofs = fem.locate_dofs_topological((self.W1, self.Q),
                                           self.domain.topology.dim - 1,
                                           facets)
        bc = fem.dirichletbc(p_zero, dofs, self.W1)
        self.add_dirichlet_bc(bc)
        return bc
        

    @staticmethod
    def build_mixed_space(domain, deg_u=2, deg_p=1):
        """Erzeugt das gemischte (velocity-pressure) Funktionsraumtuple."""
        elem_u = element("Lagrange", domain.basix_cell(), deg_u, shape=(domain.geometry.dim,))
        elem_p = element("Lagrange", domain.basix_cell(), deg_p)
        mixed_elem = mixed_element([elem_u, elem_p])

        W = fem.functionspace(domain, mixed_elem)
        W0, W1 = W.sub(0), W.sub(1)
        V, V_to_W = W0.collapse()
        Q, Q_to_W = W1.collapse()
        return W, W0, W1, V, Q, np.array(V_to_W, dtype=np.int32), np.array(Q_to_W, dtype=np.int32)

    @staticmethod
    def _read_physical_name_map(meshfile: Path) -> dict[str, int]:
        import gmsh

        meshfile = Path(meshfile).resolve(strict=True)
        gmsh.initialize()
        try:
            gmsh.open(str(meshfile))
            groups = gmsh.model.getPhysicalGroups()
            return {gmsh.model.getPhysicalName(dim, tag): tag for (dim, tag) in groups}
        finally:
            gmsh.finalize()

    @staticmethod
    def _num_dofs(space) -> int:
        return space.dofmap.index_map.size_global * space.dofmap.index_map_bs


    @staticmethod
    def normalize_regularization_mode(mode: str) -> str:
        mode_norm = mode.strip().lower()
        if mode_norm not in {"smooth", "value"}:
            raise ValueError("regularization mode must be 'smooth' or 'value'.")
        return mode_norm

    def set_regularization(self, mode: str):
        self.regularization_mode = self.normalize_regularization_mode(mode)

    def _build_solver_context(self) -> AirflowSolverConfig:
        return AirflowSolverConfig(
            domain=self.domain,
            W=self.W,
            V=self.V,
            V_to_W=self.V_to_W,
            bcs=self.bcs,
            w_measured=self.w_measured,
            measurement_ids_W=self.measurement_ids_W,
            viscosity=self.viscosity,
            weight_misfit=self.weight_misfit,
            weight_pde_res=self.weight_pde_res,
            weight_reg=self.weight_reg,
            weight_boundary=self.weight_boundary,
            regularization_mode=self.regularization_mode,
        )

    def solve_minimum_residual(self,
                               maxit: int = 10,
                               tol: float = 1e-2,
                               damping: float | None = None,
                               regularization: str | None = None,
                               verbose: bool = False):
        solver = MinimumResidualSolver(self._build_solver_context())
        result = solver.solve(
            maxit=maxit,
            tol=tol,
            damping=damping,
            regularization=regularization,
            verbose=verbose,
        )
        self.last_solver_status = solver.last_status
        self.w_final = result
        return result

    def solve_weak_penalty(self,
                           maxit: int = 10,
                           tol: float = 1e-2,
                           damping: float | None = None,
                           regularization: str | None = None,
                           verbose: bool = False):
        solver = WeakPenaltySolver(self._build_solver_context())
        result = solver.solve(
            maxit=maxit,
            tol=tol,
            damping=damping,
            regularization=regularization,
            verbose=verbose,
        )
        self.last_solver_status = solver.last_status
        self.w_final = result
        return result

    def solve_linear_least_squares(self,
                                   maxit: int = 10,
                                   tol: float = 1e-3,
                                   regularization: str | None = None,
                                   verbose: bool = False):
        solver = LinearLeastSquaresSolver(self._build_solver_context())
        result = solver.solve(
            maxit=maxit,
            tol=tol,
            regularization=regularization,
            verbose=verbose,
        )
        self.last_solver_status = solver.last_status
        self.w_final = result
        return result

    def add_dirichlet_bc(self, bc: fem.DirichletBC | list[fem.DirichletBC]):
        if isinstance(bc, list):
            self.bcs += bc
        else:
            self.bcs.append(bc)

    def set_measurements(
        self,
        measurement_ids_W: np.ndarray,
        measurement_values: np.ndarray,
        clear_existing: bool = True,
    ):
        """
        Set explicit wind measurements in mixed space W.

        Parameters
        ----------
        measurement_ids_W : np.ndarray
            Flattened W-indices for velocity components.
        measurement_values : np.ndarray
            Flattened measurement values aligned with measurement_ids_W.
        clear_existing : bool
            If True, clear all previous measurements first.
        """
        ids = np.asarray(measurement_ids_W, dtype=np.int32).reshape(-1)
        values = np.asarray(measurement_values, dtype=float).reshape(-1)

        if ids.size == 0:
            raise ValueError("measurement_ids_W is empty.")
        if ids.size != values.size:
            raise ValueError(
                f"Length mismatch: len(ids)={ids.size} != len(values)={values.size}"
            )
        if np.any(ids < 0) or np.any(ids >= self.w_measured.x.array.size):
            raise ValueError("measurement_ids_W contains out-of-bounds indices.")

        if clear_existing:
            self.w_measured.x.array[:] = 0.0

        self.w_measured.x.array[ids] = values
        self.measurement_ids_W = ids
        self.w_final = None

    def set_measurements_from_csv(
        self,
        samples_csv: str | Path,
        count: int | None = None,
        noise_std: float | None = None,
        max_xy_dist: float | None = None,
    ) -> dict[str, float]:
        """
        Load wind samples from CSV and map them to nearest velocity nodes.

        Expected CSV columns: x, y, wind_x, wind_y.

        Parameters
        ----------
        samples_csv : str | Path
            Path to sample CSV.
        count : int | None
            Optional number of rows to import from the top of the CSV. If omitted, all rows are used.
        noise_std : float | None
            Optional standard deviation for gaussian noise level to be added to the measurements.
        max_xy_dist : float | None
            Optional maximum allowed nearest-neighbor mapping distance in XY.

        Returns
        -------
        dict[str, float]
            Mapping stats (input/used samples, dropped duplicates, max distance).
        """
        return self.measurements.set_from_csv(
            samples_csv=samples_csv,
            count=count,
            noise_std=noise_std,
            max_xy_dist=max_xy_dist,
        )

    def reset_random_measurements(self, p: int, seed: int | None = None):
        """
        Creates new random measurement set overwriting self.measurement_ids_W and selfw_measured.
        Resets the solution self.w_final.
        Benötigt, dass eine ground_truth-Funktion gesetzt wurde.
        """
        self.measurements.reset_random(p=p, seed=seed)

    def set_ground_truth(self, funW: fem.Function):
        if self.ground_truth:
            warnings.warn("Overwriting ground truth data.")
        self.ground_truth = funW
            
    def set_weights(self, kin_v: float | None = None, 
                          misfit: float | None = None, 
                          pde_err: float | None = None, 
                          reg: float | None = None,
                          boundary: float | None = None):
        if kin_v is not None:
            self.viscosity = kin_v
        if misfit is not None:
            self.weight_misfit = misfit
        if pde_err is not None:
            self.weight_pde_res = pde_err
        if reg is not None:
            self.weight_reg = reg
        if boundary is not None:
            self.weight_boundary = boundary
        
    def get_measurement_coordinates(self) -> np.ndarray:
        """Rekonstruiere Messpunkt-Koordinaten aus measurement_ids_W."""
        return self.measurements.coordinates()
