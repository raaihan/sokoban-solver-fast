// sokoban-solver-fast: a push-optimal Sokoban solver for make.fun.
// 
// A level is a hallway with independent rooms off it (a box pushed into the
// hallway can never come back, so each room's boxes stay in their room and
// the level's optimum is the sum of the rooms'). Each room is solved on its
// own with A* over pushes -- a state is the boxes plus the player's reachable
// region -- with an admissible min-assignment heuristic (boxes matched to
// goals by shortest floor distance), dead-square pruning and frozen 2x2
// pruning, then the room solutions are stitched with player walks through
// the hallway. Deterministic: fixed direction order, heap ties broken by
// insertion sequence, budgets by state counts only -- no clock, no random.
//
// Plain JavaScript in a .ts file (erasable syntax only), Node built-ins only.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Push/search budgets, in states -- never the clock, so two runs agree.
const MAX_STATES_ROOM = 1_000_000;
const MAX_STATES_WHOLE = 400_000;

// Directions: [dx, dy, walk letter, push letter], fixed order everywhere.
const DIRS = [
  [0, -1, 'u', 'U'],
  [0, 1, 'd', 'D'],
  [-1, 0, 'l', 'L'],
  [1, 0, 'r', 'R'],
];

/** Parse XSB text into a board. Walls pad short rows and the outer frame. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      if (x === 0 || y === 0 || x === w - 1 || y === h - 1) wall[y * w + x] = 1;
    }
  }
  return { w, h, wall, goals, boxes: boxes.sort((a, b) => a - b), player };
}

/** Squares from which a box can be pushed onto some goal (pull-back from goals). */
function liveSquares(level) {
  const { w, wall, goals } = level;
  const live = new Uint8Array(wall.length);
  const queue = [...goals];
  for (const g of goals) live[g] = 1;
  for (let qi = 0; qi < queue.length; qi++) {
    const x = queue[qi];
    for (const [dx, dy] of DIRS) {
      const d = dy * w + dx;
      const y = x - d;
      if (!wall[y] && !wall[y - d] && !live[y]) {
        live[y] = 1;
        queue.push(y);
      }
    }
  }
  return live;
}

/** BFS distances from every floor cell to each goal (ignoring boxes). */
function goalDistances(level) {
  const { w, wall, goals } = level;
  const n = goals.length;
  const dist = new Int32Array(wall.length * n).fill(-1);
  for (let gi = 0; gi < n; gi++) {
    const g = goals[gi];
    const queue = [g];
    dist[gi * wall.length + g] = 0;
    for (let qi = 0; qi < queue.length; qi++) {
      const p = queue[qi];
      const d0 = dist[gi * wall.length + p];
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && dist[gi * wall.length + q] < 0) {
          dist[gi * wall.length + q] = d0 + 1;
          queue.push(q);
        }
      }
    }
  }
  return dist;
}

/**
 * The level's rooms: cut the dead squares of the player's row (the hallway)
 * out of the floor; each remaining connected part holding a box is a room.
 * Null when a box starts in the hallway, a goal falls outside every part,
 * or a part's boxes and goals differ in count.
 */
function roomsOf(level) {
  const { w, wall, goals, boxes, player } = level;
  const live = liveSquares(level);
  const row = Math.floor(player / w);
  const cut = new Uint8Array(wall.length);
  for (let i = row * w; i < (row + 1) * w; i++) {
    if (!wall[i] && !live[i]) cut[i] = 1;
  }
  if (boxes.some((b) => cut[b])) return null;
  const part = new Int32Array(wall.length).fill(-1);
  const rooms = [];
  for (let s = 0; s < wall.length; s++) {
    if (wall[s] || cut[s] || part[s] >= 0) continue;
    const squares = [s];
    part[s] = rooms.length;
    for (let i = 0; i < squares.length; i++) {
      for (const [dx, dy] of DIRS) {
        const q = squares[i] + dy * w + dx;
        if (!wall[q] && !cut[q] && part[q] < 0) {
          part[q] = rooms.length;
          squares.push(q);
        }
      }
    }
    rooms.push({ squares, boxes: [], goals: [] });
  }
  for (const b of boxes) rooms[part[b]].boxes.push(b);
  for (const g of goals) if (part[g] >= 0) rooms[part[g]].goals.push(g);
  if (goals.some((g) => part[g] < 0)) return null;
  if (rooms.some((r) => r.boxes.length !== r.goals.length)) return null;
  return rooms.filter((r) => r.boxes.length > 0);
}

/** Shortest walk from `from` to `to` around walls and boxes (lower letters). */
function walkTo(w, wall, occupied, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(wall.length).fill(-1);
  const how = new Array(wall.length);
  const queue = [from];
  prev[from] = from;
  for (let qi = 0; qi < queue.length; qi++) {
    const p = queue[qi];
    if (p === to) break;
    for (const [dx, dy, letter] of DIRS) {
      const q = p + dy * w + dx;
      if (wall[q] || occupied[q] || prev[q] !== -1) continue;
      prev[q] = p;
      how[q] = letter;
      queue.push(q);
    }
  }
  if (prev[to] === -1) return null;
  let s = '';
  for (let p = to; p !== from; p = prev[p]) s = how[p] + s;
  return s;
}

/** Small binary heap ordered by (f, h, sequence): fully deterministic. */
function makeHeap() {
  const a = [];
  const less = (x, y) =>
    x[0] < y[0] || (x[0] === y[0] && (x[1] < y[1] || (x[1] === y[1] && x[2] < y[2])));
  return {
    get size() {
      return a.length;
    },
    push(e) {
      let i = a.length;
      a.push(e);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!less(a[i], a[p])) break;
        const t = a[i];
        a[i] = a[p];
        a[p] = t;
        i = p;
      }
    },
    pop() {
      const top = a[0];
      const last = a.pop();
      if (a.length > 0) {
        a[0] = last;
        for (let i = 0; ; ) {
          let c = 2 * i + 1;
          if (c >= a.length) break;
          if (c + 1 < a.length && less(a[c + 1], a[c])) c++;
          if (!less(a[c], a[i])) break;
          const t = a[i];
          a[i] = a[c];
          a[c] = t;
          i = c;
        }
      }
      return top;
    },
  };
}

/**
 * Push-optimal solution of `sub` (a board with its own walls/goals/boxes/
 * player): {pushes, moves, endPlayer}, or null when the state budget runs
 * out or the board has no solution. A* over pushes; a state is the boxes
 * plus the player's reachable region (named by its lowest square).
 */
function solvePushAStar(sub, maxStates) {
  const { w, wall, goals } = sub;
  const size = wall.length;
  const n = goals.length;
  const goal = new Uint8Array(size);
  for (const g of goals) goal[g] = 1;
  const live = liveSquares(sub);
  const dist = goalDistances(sub);
  const solved = (bs) => bs.every((b) => goal[b]);
  if (sub.boxes.some((b) => !live[b])) return null;
  if (solved(sub.boxes)) return { pushes: 0, moves: '', endPlayer: sub.player };

  // State key: region-low and boxes packed base-`size` (exact while the
  // packing stays under 2^53); a string key when it would not.
  const numeric = Math.pow(size, n + 1) < 2 ** 53;
  const keyOf = (bs, low) => {
    if (!numeric) return `${low}:${bs.join(',')}`;
    let k = low;
    for (const b of bs) k = k * size + b;
    return k;
  };
  const boxKeyOf = (bs) => {
    if (!numeric) return bs.join(',');
    let k = 0;
    for (const b of bs) k = k * size + b;
    return k;
  };

  // Min-assignment heuristic: boxes matched to goals by floor distance,
  // brute-forced over the goal permutations (n <= 4 here); memoized by box
  // configuration. Infinity when no finite matching exists (dead state).
  const hMemo = new Map();
  const heuristic = (bs) => {
    const bk = boxKeyOf(bs);
    const hit = hMemo.get(bk);
    if (hit !== undefined) return hit;
    let best = Infinity;
    const rec = (i, used, cost) => {
      if (cost >= best) return;
      if (i === n) {
        best = cost;
        return;
      }
      for (let j = 0; j < n; j++) {
        if (used & (1 << j)) continue;
        const d = dist[j * size + bs[i]];
        if (d < 0) continue;
        rec(i + 1, used | (1 << j), cost + d);
      }
    };
    rec(0, 0, 0);
    hMemo.set(bk, best);
    return best;
  };

  const occupied = new Uint8Array(size);
  const seen = new Int32Array(size);
  const mark = new Int32Array(size);
  const stack = new Int32Array(size);
  let stamp = 0;
  let markStamp = 0;
  // Reachable region from `from` (boxes in `occupied` block), marked in
  // `mark` with a fresh stamp (a separate array from the expansion's
  // `seen`, so computing a successor's region never clobbers the region
  // the expansion is testing pushes against); returns its lowest square.
  const regionLow = (from) => {
    const st = ++markStamp;
    let low = from;
    let m = 0;
    stack[m++] = from;
    mark[from] = st;
    while (m > 0) {
      const p = stack[--m];
      if (p < low) low = p;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && !occupied[q] && mark[q] !== st) {
          mark[q] = st;
          stack[m++] = q;
        }
      }
    }
    return low;
  };
  // Whether the box just pushed to `t` sits in a frozen 2x2 block of walls
  // and boxes with a box off its goal.
  const frozenBlock = (t) => {
    for (const a of [t, t - 1, t - w, t - w - 1]) {
      const b = a + 1;
      const c = a + w;
      const d = c + 1;
      if (!(wall[a] || occupied[a])) continue;
      if (!(wall[b] || occupied[b])) continue;
      if (!(wall[c] || occupied[c])) continue;
      if (!(wall[d] || occupied[d])) continue;
      if (
        (occupied[a] && !goal[a]) ||
        (occupied[b] && !goal[b]) ||
        (occupied[c] && !goal[c]) ||
        (occupied[d] && !goal[d])
      ) {
        return true;
      }
    }
    return false;
  };

  const sBoxes = [sub.boxes.slice()];
  const sPlayer = [sub.player];
  const sParent = [-1];
  const sFrom = [-1];
  const sTo = [-1];
  const sDir = [-1];
  const sG = [0];
  const best = new Map();
  for (const b of sub.boxes) occupied[b] = 1;
  const startLow = regionLow(sub.player);
  for (const b of sub.boxes) occupied[b] = 0;
  const h0 = heuristic(sub.boxes);
  if (h0 === Infinity) return null;
  best.set(keyOf(sub.boxes, startLow), 0);
  const heap = makeHeap();
  let seq = 0;
  heap.push([h0, h0, seq++, 0]);

  let found = -1;
  while (heap.size > 0) {
    const [f, h, , id] = heap.pop();
    const g = sG[id];
    if (f !== g + h) continue; // stale heap entry
    const boxes = sBoxes[id];
    if (solved(boxes)) {
      found = id;
      break;
    }
    if (sBoxes.length >= maxStates) return null;
    for (const b of boxes) occupied[b] = 1;
    const st = ++stamp;
    // Region of this state, from the stored player square.
    let low = sPlayer[id];
    let m = 0;
    stack[m++] = sPlayer[id];
    seen[sPlayer[id]] = st;
    while (m > 0) {
      const p = stack[--m];
      if (p < low) low = p;
      for (const [dx, dy] of DIRS) {
        const q = p + dy * w + dx;
        if (!wall[q] && !occupied[q] && seen[q] !== st) {
          seen[q] = st;
          stack[m++] = q;
        }
      }
    }
    for (let bi = 0; bi < boxes.length; bi++) {
      const b = boxes[bi];
      for (let di = 0; di < 4; di++) {
        const [dx, dy] = DIRS[di];
        const d = dy * w + dx;
        const t = b + d;
        if (seen[b - d] !== st || wall[t] || occupied[t] || !live[t]) continue;
        occupied[b] = 0;
        occupied[t] = 1;
        if (frozenBlock(t)) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const next = boxes.slice();
        next[bi] = t;
        next.sort((x, y) => x - y);
        const nextLow = regionLow(b);
        const key = keyOf(next, nextLow);
        const g2 = g + 1;
        const prevBest = best.get(key);
        if (prevBest !== undefined && prevBest <= g2) {
          occupied[t] = 0;
          occupied[b] = 1;
          continue;
        }
        const h2 = heuristic(next);
        occupied[t] = 0;
        occupied[b] = 1;
        if (h2 === Infinity) continue;
        best.set(key, g2);
        const id2 = sBoxes.length;
        sBoxes.push(next);
        sPlayer.push(b);
        sParent.push(id);
        sFrom.push(b);
        sTo.push(t);
        sDir.push(di);
        sG.push(g2);
        heap.push([g2 + h2, h2, seq++, id2]);
      }
    }
    for (const b of boxes) occupied[b] = 0;
  }
  if (found < 0) return null;

  // Rebuild the push sequence, then the moves: walk to each push square,
  // then push. Forward replay keeps player/boxes in step with the states.
  const path = [];
  for (let s = found; s > 0; s = sParent[s]) path.push(s);
  path.reverse();
  let moves = '';
  let player = sub.player;
  let boxes = sub.boxes.slice();
  for (const s of path) {
    const from = sFrom[s];
    const to = sTo[s];
    const [dx, dy, , pushLetter] = DIRS[sDir[s]];
    const d = dy * w + dx;
    for (const b of boxes) occupied[b] = 1;
    const walk = walkTo(w, wall, occupied, player, from - d);
    for (const b of boxes) occupied[b] = 0;
    if (walk === null) return null;
    moves += walk + pushLetter;
    player = from;
    const at = boxes.indexOf(from);
    boxes[at] = to;
    boxes.sort((a, b) => a - b);
  }
  return { pushes: path.length, moves, endPlayer: player };
}

/** The level solved room by room: {pushes, moves}, or null. */
function solveByRooms(level) {
  const rooms = roomsOf(level);
  if (rooms === null) {
    const whole = solvePushAStar(
      { w: level.w, h: level.h, wall: level.wall, goals: level.goals, boxes: level.boxes, player: level.player },
      MAX_STATES_WHOLE,
    );
    return whole === null ? null : { pushes: whole.pushes, moves: whole.moves };
  }
  let player = level.player;
  let pushes = 0;
  let moves = '';
  for (const room of rooms) {
    const wall = level.wall.slice();
    for (const other of rooms) {
      if (other === room) continue;
      // A room already solved has its boxes on its goals; one still to come
      // has them where they started. Either way those cells are walls here.
      for (const b of other.done ? other.goals : other.boxes) wall[b] = 1;
    }
    const sub = {
      w: level.w,
      h: level.h,
      wall,
      goals: room.goals.slice().sort((a, b) => a - b),
      boxes: room.boxes.slice().sort((a, b) => a - b),
      player,
    };
    const solved = solvePushAStar(sub, MAX_STATES_ROOM);
    if (solved === null) return null;
    pushes += solved.pushes;
    moves += solved.moves;
    player = solved.endPlayer;
    room.done = true;
  }
  return { pushes, moves };
}

/** The moves for one level's XSB text; '' when it cannot be solved. */
function solveLevel(text) {
  const level = parseLevel(text);
  if (level.boxes.every((b, i) => level.goals.includes(b))) return '';
  const solved = solveByRooms(level);
  return solved === null ? '' : solved.moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
