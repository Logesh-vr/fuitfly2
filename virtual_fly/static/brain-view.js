import * as THREE from './three/three.module.js';

function createBrainView(prefix) {
const element = id => document.getElementById(prefix + id.slice(5));
const host = element('brainViewport');
const info = element('brainInfo');
const inspector = element('brainInspector');
const filter = element('brainFilter');
let geometryData = null, latestState = {}, counts = [], points, lines;
let viewYaw = .35, viewPitch = .2, viewDistance = 6.5, drag = null;
let lastBrainTick = -1, selected = -1, activityScale = 250;
let scene, camera, renderer, raycaster;

function visible(index) {
  const kind = geometryData.neurons[index].kind;
  return filter.value === 'all' || filter.value === kind ||
    (filter.value === 'active' && (counts[index] || 0) > 0);
}

function inspectNeuron(index) {
  selected = index;
  if (index < 0 || !geometryData) {
    inspector.textContent = 'Hover or click a neuron to inspect its identity and measured activity.';
    return;
  }
  const neuron = geometryData.neurons[index];
  const n = counts[index] || 0;
  const dt = latestState.brain_window_s || .01;
  inspector.textContent = `FlyWire ${neuron.id} · ${neuron.group} · ${neuron.side} · ` +
    `${n} spikes / ${(dt * 1000).toFixed(0)} ms (${(n / dt).toFixed(0)} Hz). ` +
    (latestState.controls?.mode === 'manual' ? 'Brain frozen in manual mode.' : 'Latest completed neural interval.');
}

function recolor() {
  if (!points || !geometryData) return;
  const colors = points.geometry.attributes.color;
  const sizes = points.geometry.attributes.aSize;
  const palette = {sensory: new THREE.Color('#9ee8bc'), output: new THREE.Color('#f4b078'), internal: new THREE.Color('#82bcfa')};
  const white = new THREE.Color('#fff9de');
  const color = new THREE.Color();
  const dt = latestState.brain_window_s || .01;
  let active = 0, spikes = 0;
  geometryData.neurons.forEach((neuron, index) => {
    const count = counts[index] || 0;
    if (count > 0) active++;
    spikes += count;
    const intensity = Math.sqrt(Math.min(1, count / dt / activityScale));
    color.copy(palette[neuron.kind]);
    if (count) color.lerp(white, intensity * .8);
    else color.multiplyScalar(neuron.kind === 'internal' ? .28 : .55);
    colors.setXYZ(index, color.r, color.g, color.b);
    sizes.setX(index, visible(index) ? (count ? 1.8 : 1) : 0);
  });
  colors.needsUpdate = sizes.needsUpdate = true;
  const edgeColors = lines.geometry.attributes.color;
  geometryData.edges.forEach(([a, b, weight], i) => {
    const glow = counts[a] > 0;
    color.set(weight < 0 ? '#e995aa' : '#73aac7');
    color.multiplyScalar(visible(a) && visible(b) ? (glow ? .8 : .07) : 0);
    edgeColors.setXYZ(i * 2, color.r, color.g, color.b);
    edgeColors.setXYZ(i * 2 + 1, color.r, color.g, color.b);
  });
  edgeColors.needsUpdate = true;
  const mode = latestState.controls?.mode === 'manual' ? 'FROZEN · manual mode' :
    latestState.status === 'computing' ? 'Last completed step · computing next' :
    latestState.controls?.paused ? 'PAUSED · last completed step' : 'LIVE · step snapshots';
  element('brainSnapshot').textContent = mode;
  info.textContent = `${geometryData.displayed_neurons.toLocaleString()} of ${geometryData.total_neurons.toLocaleString()} neurons · ` +
    `${active.toLocaleString()} active · ${spikes.toLocaleString()} spikes in the displayed sample · neural time ${(latestState.brain_time_s || 0).toFixed(2)} s`;
  inspectNeuron(selected);
}

function createGeometry(data) {
  geometryData = data;
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.positions, 3));
  geometry.setAttribute('color', new THREE.Float32BufferAttribute(new Float32Array(data.displayed_neurons * 3), 3));
  geometry.setAttribute('aSize', new THREE.Float32BufferAttribute(new Float32Array(data.displayed_neurons), 1));
  const material = new THREE.ShaderMaterial({
    uniforms: {pointSize: {value: 3 * Math.min(devicePixelRatio, 2)}},
    vertexShader: `attribute vec3 color; attribute float aSize; varying vec3 vColor; varying float vSize;
      uniform float pointSize;
      void main(){vColor=color; vSize=aSize; vec4 mv=modelViewMatrix*vec4(position,1.);
        gl_Position=projectionMatrix*mv; gl_PointSize=aSize*pointSize*(6./max(1.,-mv.z));}`,
    fragmentShader: `varying vec3 vColor; varying float vSize;
      void main(){float r=length(gl_PointCoord-vec2(.5)); if(r>.5||vSize<.1)discard;
        gl_FragColor=vec4(vColor,1.-smoothstep(.22,.5,r));}`,
    transparent: true, depthWrite: false,
  });
  points = new THREE.Points(geometry, material);
  scene.add(points);
  const linePositions = new Float32Array(data.edges.length * 6);
  data.edges.forEach(([a, b], i) => {
    linePositions.set(data.positions.slice(a * 3, a * 3 + 3), i * 6);
    linePositions.set(data.positions.slice(b * 3, b * 3 + 3), i * 6 + 3);
  });
  const edgeGeometry = new THREE.BufferGeometry();
  edgeGeometry.setAttribute('position', new THREE.BufferAttribute(linePositions, 3));
  edgeGeometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(linePositions.length), 3));
  lines = new THREE.LineSegments(edgeGeometry, new THREE.LineBasicMaterial({vertexColors: true, transparent: true, opacity: .3, depthWrite: false}));
  scene.add(lines);
  element('brainEdgeLabel').textContent = `Show ${data.displayed_connections.toLocaleString()} sampled connections`;
  recolor();
}

const update = function(state) {
  latestState = state;
  const tick = state.brain_tick ?? 0;
  if (tick !== lastBrainTick || counts.length === 0) {
    counts = state.brain_view_counts || [];
    lastBrainTick = tick;
  }
  recolor();
};

filter.addEventListener('change', recolor);
element('brainScale').addEventListener('change', event => {activityScale = Number(event.target.value); recolor();});
element('brainEdges').addEventListener('change', event => {if (lines) lines.visible = event.target.checked;});
element('brainReset').addEventListener('click', () => {viewYaw = .35; viewPitch = .2; viewDistance = 6.5;});

try {
  scene = new THREE.Scene();
  scene.background = new THREE.Color('#080f18');
  camera = new THREE.PerspectiveCamera(45, 1, .1, 100);
  renderer = new THREE.WebGLRenderer({antialias: true, alpha: false});
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  host.appendChild(renderer.domElement);
  raycaster = new THREE.Raycaster();
  raycaster.params.Points.threshold = .035;
  const observer = new ResizeObserver(() => {
    const width = host.clientWidth, height = host.clientHeight;
    if (!width || !height) return;
    renderer.setSize(width, height);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
  });
  observer.observe(host);
  host.addEventListener('pointerdown', event => {drag = [event.clientX, event.clientY];host.setPointerCapture(event.pointerId);});
  host.addEventListener('pointerup', () => {drag = null;});
  host.addEventListener('pointercancel', () => {drag = null;});
  host.addEventListener('pointermove', event => {
    if (drag) {
      viewYaw -= (event.clientX - drag[0]) * .008;
      viewPitch = Math.max(-1.35, Math.min(1.35, viewPitch + (event.clientY - drag[1]) * .008));
      drag = [event.clientX, event.clientY];
    } else if (points) {
      const rect = host.getBoundingClientRect();
      raycaster.setFromCamera(new THREE.Vector2((event.clientX - rect.left) / rect.width * 2 - 1,
        -(event.clientY - rect.top) / rect.height * 2 + 1), camera);
      const hit = raycaster.intersectObject(points).find(item => visible(item.index));
      inspectNeuron(hit ? hit.index : -1);
    }
  });
  host.addEventListener('wheel', event => {event.preventDefault();viewDistance = Math.max(3.2, Math.min(16, viewDistance * Math.exp(event.deltaY * .001)));}, {passive: false});
  function animate() {
    requestAnimationFrame(animate);
    if (document.hidden || element('brainCard').hidden) return;
    camera.position.set(viewDistance * Math.sin(viewYaw) * Math.cos(viewPitch),
      viewDistance * Math.sin(viewPitch), viewDistance * Math.cos(viewYaw) * Math.cos(viewPitch));
    camera.lookAt(0, 0, 0);
    renderer.render(scene, camera);
  }
  animate();
  async function loadGeometry() {
    try {
      const response = await fetch('/api/brain_geometry');
      if (response.status === 202) {setTimeout(loadGeometry, 2000);return;}
      if (!response.ok) throw Error('Restart the live launcher to enable the 3D brain data endpoint.');
      createGeometry(await response.json());
    } catch (error) {info.textContent = error.message;}
  }
  loadGeometry();
} catch (error) {
  info.textContent = `3D renderer unavailable: ${error.message}. Enable browser graphics acceleration to use this view.`;
}

return update;
}
const outerUpdate = createBrainView('brain');
const innerCard = document.getElementById('brainCard').cloneNode(true);
innerCard.querySelector('canvas')?.remove();
innerCard.id = 'innerBrainCard';
innerCard.querySelectorAll('[id]').forEach(node => {node.id = 'innerBrain' + node.id.slice(5);});
innerCard.querySelector('.cardtitle').firstChild.textContent = "Inner fly's brain activity in 3D ";
innerCard.hidden = true;
document.getElementById('computerCard').after(innerCard);
const innerUpdate = createBrainView('innerBrain');
window.updateBrainView = state => {
  outerUpdate(state);
  innerCard.hidden = !state.inner;
  if (state.inner) innerUpdate({...state.inner, status: state.status,
    brain_window_s: state.brain_window_s,
    controls: {mode: 'brain', paused: state.controls?.paused || state.controls?.inner_paused}});
};