"""Process isolation keeps the two Brian2 clocks and random streams independent."""
import multiprocessing as mp
import traceback


def _serve(connection, config, seed, indices):
    try:
        from .brain import ConnectomeBrain
        brain = ConnectomeBrain(config, seed)
        connection.send({"ready": True, "neurons": len(brain.ids),
                         "synapses": brain.synapse_count})
        while mp.parent_process().is_alive():
            if not connection.poll(.25):
                continue
            request = connection.recv()
            if request is None:
                break
            activity = brain.step(request["stimulation"], request["dt"])
            connection.send({"rates": activity.rates_hz,
                             "counts": brain.last_spike_counts[indices].tolist()})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()


class InnerBrain:
    def __init__(self, config, seed, indices, stop_event):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        self.stop_event = stop_event
        self.process = context.Process(target=_serve,
            args=(child, {**config, "record_spikes": False}, seed, indices),
            name="InnerFlyConnectome", daemon=True)
        self.process.start()
        child.close()
        try:
            self.metadata = self.receive()
        except BaseException:
            self.close()
            raise

    def receive(self):
        while not self.connection.poll(.25):
            if self.stop_event.is_set():
                raise InterruptedError("Session ended while inner brain was computing")
            if not self.process.is_alive():
                raise RuntimeError("Inner brain process exited")
        response = self.connection.recv()
        if "error" in response:
            raise RuntimeError(response["error"])
        return response

    def step(self, stimulation, dt):
        self.connection.send({"stimulation": stimulation, "dt": dt})
        return self.receive()

    def close(self):
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError, OSError):
                pass
            self.process.join(timeout=1)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(timeout=2)
        self.connection.close()
