"""Persistent single-elimination tournaments with an optional LAN host."""
from __future__ import annotations

import json
import random
import copy
import socket
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

STATE_PATH = Path(__file__).with_name("tournament_state.json")
PORT = 8765
DISCOVERY_PORT = 8766
CLAIM_LEASE_SECONDS = 45.0


def _atomic_save(path: Path, state: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(path)


class Tournament:
    def __init__(self, path: Path = STATE_PATH):
        self.path, self.lock = path, threading.RLock()
        self.state = self._load()
        self._history = []
        # Safely upgrade an old, untouched power-of-two bracket to the compact
        # layout. Once play started we keep its exact pairings and use the
        # legacy routing fallback in _put_winner.
        if self.state.get("id") and self.state.get("matches") and (
            int(self.state.get("layout_version") or 0) < 3
        ) and not self._play_started():
            self.state["size"]=len(self.state.get("players") or [])
            self.state["matches"]=self._build_compact_matches(list(self.state.get("players") or []))
            self.state["layout_version"]=3
            self._advance_byes(); self.save()

    def _load(self):
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def _play_started(self):
        return any(
            m.get("status") == "playing"
            or (m.get("status") == "finished" and len([p for p in m.get("players", []) if p]) == 2)
            for m in self.state.get("matches", [])
        )

    def snapshot(self):
        with self.lock:
            if self._expire_stale_claims():
                self.save()
            snap = json.loads(json.dumps(self.state))
            if snap:
                snap["can_undo"] = bool(self._history)
            return snap

    def _expire_stale_claims(self):
        """Release matches whose board stopped sending its lease heartbeat."""
        changed=False; now=time.time()
        for match in self.state.get("matches",[]):
            if match.get("status")!="playing":
                continue
            claimed_at=float(match.get("claimed_at") or 0.0)
            if claimed_at and now-claimed_at > CLAIM_LEASE_SECONDS:
                match["status"],match["board"]="ready",None
                match.pop("claimed_at",None); changed=True
        return changed

    def create(self, names, game="501", *, shuffle=False, rules=None):
        clean = [str(x).strip().upper()[:16] for x in names if str(x).strip()]
        if not 2 <= len(clean) <= 16:
            raise ValueError("Turnir podržava od 2 do 16 igrača")
        if len(set(clean)) != len(clean):
            raise ValueError("Svaki igrač mora imati jedinstveno ime")
        if shuffle:
            random.shuffle(clean)
        matches = self._build_compact_matches(clean)
        self._history = []
        self.state={"id":uuid.uuid4().hex[:8],"name":"SMARTDARTS TURNIR",
                    "game":game,"rules":dict(rules or {}),"status":"active","players":clean,
                    "size":len(clean),"matches":matches,"winner":None,"layout_version":3,
                    "updated":time.time()}
        self._advance_byes(); self.save(); return self.snapshot()

    def close(self):
        with self.lock:
            self._history = []
            self.state = {}
            _atomic_save(self.path, self.state)
            return {}

    @staticmethod
    def _match(rnd, slot, a=None, b=None):
        return {"id":f"R{rnd}M{slot+1}","round":rnd,"slot":slot,
                "players":[a,b],"status":"ready" if a and b else "waiting",
                "board":None,"winner":None}

    def _build_compact_matches(self, players):
        """Build a compact bracket with only the necessary preliminary games.

        Eleven players, for example, need three preliminary games and five
        byes to produce an eight-player quarter-final round.
        """
        count=len(players)
        base=1 << (count.bit_length()-1)
        prelim=max(0,count-base)
        matches=[]; previous=[]; rnd=1; player_index=0

        if prelim == 0:
            for slot in range(count//2):
                a,b=players[player_index],players[player_index+1]; player_index += 2
                m=self._match(rnd,slot,a,b); m["expected"]=2
                matches.append(m); previous.append(m)
        else:
            # Spread the exact number of preliminary games over `base`
            # source slots. This also handles counts such as 7 (3 games,
            # 1 bye), where a simple bye/game alternation is impossible.
            accumulator=0
            for slot in range(base):
                accumulator += prelim
                is_pair = accumulator >= base
                if is_pair:
                    accumulator -= base
                    m=self._match(rnd,slot,players[player_index],players[player_index+1])
                    m["expected"]=2; player_index += 2
                else:
                    m=self._match(rnd,slot,players[player_index],None)
                    m["expected"]=1; player_index += 1
                matches.append(m); previous.append(m)
        while len(previous) > 1:
            rnd += 1; current=[]
            for i in range(0, len(previous), 2):
                expected=2 if i+1 < len(previous) else 1
                m=self._match(rnd, i//2); m["expected"]=expected
                matches.append(m); current.append(m)
                previous[i]["next_id"],previous[i]["next_pos"]=m["id"],0
                if expected == 2:
                    previous[i+1]["next_id"],previous[i+1]["next_pos"]=m["id"],1
            previous=current
        return matches

    def save(self):
        self.state["updated"]=time.time(); _atomic_save(self.path,self.state)

    def _next(self, match):
        return int(match["round"])+1, int(match["slot"])//2, int(match["slot"])%2

    def _put_winner(self, match, winner):
        target_id=match.get("next_id")
        if not target_id:
            if "expected" not in match:
                rnd,slot,pos=self._next(match)
                if 2**rnd <= int(self.state.get("size") or 0):
                    mid=f"R{rnd}M{slot+1}"
                    nxt=next((m for m in self.state["matches"] if m["id"]==mid),None)
                    if nxt is None:
                        nxt=self._match(rnd,slot); self.state["matches"].append(nxt)
                    nxt["players"][pos]=winner
                    if all(nxt["players"]): nxt["status"]="ready"
                    return
            self.state["winner"],self.state["status"]=winner,"finished"; return
        nxt=next(m for m in self.state["matches"] if m["id"]==target_id)
        sources=sorted(
            (m for m in self.state["matches"] if m.get("next_id")==target_id),
            key=lambda m:int(m.get("next_pos") or 0),
        )
        # Keep the next card empty until both feeder branches have a winner.
        # This leaves the completed match visible with its loser crossed out
        # and prevents a winner from appearing alone on several later cards.
        if not sources or not all(m.get("status")=="finished" and m.get("winner") for m in sources):
            nxt["players"]=[None,None]; nxt["status"]="waiting"
            return
        nxt["players"]=[None,None]
        for source in sources:
            nxt["players"][int(source.get("next_pos") or 0)]=source.get("winner")
        present=[p for p in nxt["players"] if p]
        if len(present) >= int(nxt.get("expected") or 2):
            nxt["status"]="ready"

    def _advance_byes(self):
        changed=True
        while changed:
            changed=False
            for m in list(self.state.get("matches",[])):
                p=[x for x in m["players"] if x]
                if m["status"]=="waiting" and len(p)>=int(m.get("expected") or 2):
                    m["winner"],m["status"],m["propagated"]=p[0],"finished",False
                if m["status"]=="finished" and m.get("winner") and not m.get("propagated"):
                    m["propagated"]=True; self._put_winner(m,m["winner"]); changed=True

    def claim(self, match_id, board):
        with self.lock:
            m=next(m for m in self.state["matches"] if m["id"]==match_id)
            real_players=[p for p in m.get("players",[]) if p and str(p).strip().upper()!="NONE"]
            if len(real_players)!=2:
                m["status"]="waiting"; m["board"]=None; self.save()
                raise ValueError("Meč se može pokrenuti tek kada ima dva igrača")
            if m["status"]!="ready": raise ValueError("Meč više nije slobodan")
            busy_players={
                player
                for other in self.state.get("matches",[])
                if other.get("id")!=match_id and other.get("status")=="playing"
                for player in other.get("players",[])
                if player
            }
            if any(player in busy_players for player in real_players):
                raise ValueError("Jedan od igrača već igra na drugoj ploči")
            m["status"],m["board"]="playing",str(board); m["claimed_at"]=time.time()
            self.save(); return self.snapshot()

    def heartbeat(self, match_id, board):
        with self.lock:
            m=next(m for m in self.state["matches"] if m["id"]==match_id)
            if m.get("status")!="playing" or m.get("board")!=str(board):
                raise ValueError("Rezervacija meča više nije aktivna")
            m["claimed_at"]=time.time()
            # Lease timestamps are transport bookkeeping, not a bracket
            # change. Persist them without changing the public UI revision.
            _atomic_save(self.path,self.state); return self.snapshot()

    def cancel(self, match_id, board):
        with self.lock:
            m=next(m for m in self.state["matches"] if m["id"]==match_id)
            if m["status"]=="playing" and m["board"]==str(board):
                m["status"],m["board"]="ready",None
                m.pop("claimed_at",None); self.save()
            return self.snapshot()

    def finish(self, match_id, winner, board):
        with self.lock:
            m=next(m for m in self.state["matches"] if m["id"]==match_id)
            if m["status"]!="playing" or m["board"]!=str(board) or winner not in m["players"]:
                raise ValueError("Nevažeći rezultat meča")
            m["status"],m["winner"],m["board"],m["propagated"]="finished",winner,None,False
            m.pop("claimed_at",None)
            self._advance_byes(); self.save(); return self.snapshot()

    def ready_matches(self):
        return [m for m in self.snapshot().get("matches",[]) if m["status"]=="ready"
                and len([p for p in m.get("players",[]) if p and str(p).strip().upper()!="NONE"])==2]

    def _remember(self):
        self._history.append(copy.deepcopy(self.state))
        if len(self._history) > 50:
            self._history = self._history[-50:]

    def can_undo(self):
        return bool(self._history)

    def undo(self):
        with self.lock:
            if not self._history:
                return self.snapshot()
            self.state = self._history.pop()
            self.save(); return self.snapshot()

    def shuffle(self):
        with self.lock:
            if self._play_started():
                raise ValueError("Parovi se više ne mogu mijenjati nakon početka turnira")
            self._remember()
            players = list(self.state.get("players") or [])
            random.shuffle(players)
            self.state["matches"] = self._build_compact_matches(players)
            self.state["layout_version"] = 3
            self.state["winner"], self.state["status"] = None, "active"
            self._advance_byes(); self.save(); return self.snapshot()

    def rename(self, old_name, new_name):
        with self.lock:
            old = str(old_name or "").strip().upper()
            new = str(new_name or "").strip().upper()[:16]
            players = list(self.state.get("players") or [])
            if old not in players or not new or (new in players and new != old):
                raise ValueError("Ime igrača nije valjano ili već postoji")
            if self._play_started():
                raise ValueError("Imena se više ne mogu mijenjati nakon početka turnira")
            self._remember()
            self.state["players"] = [new if p == old else p for p in players]
            for match in self.state.get("matches", []):
                match["players"] = [new if p == old else p for p in match.get("players", [])]
            self.save(); return self.snapshot()

    def assign(self, match_id, player_a, player_b):
        """Edit one first-round pair by swapping only the selected slots."""
        with self.lock:
            if not self.state.get("id"):
                raise ValueError("Bracket još nije postavljen")
            if self._play_started():
                raise ValueError("Parovi se više ne mogu mijenjati nakon početka turnira")
            a, b = str(player_a or "").strip().upper(), str(player_b or "").strip().upper()
            players = list(self.state.get("players") or [])
            if not a or not b or a == b or a not in players or b not in players:
                raise ValueError("Odaberite dva različita igrača")
            first = sorted(
                (m for m in self.state.get("matches", []) if int(m.get("round") or 0) == 1),
                key=lambda m: int(m.get("slot") or 0),
            )
            target = next((m for m in first if m.get("id") == match_id), None)
            if target is None:
                raise ValueError("Možete uređivati samo parove prve runde")
            self._remember()

            # Keep references to every occupied first-round slot. Swapping a
            # selection into the target also moves the displaced player into
            # the selected player's old slot, without rebuilding other pairs.
            slots=[]
            for match in first:
                for pos, name in enumerate(match.get("players") or []):
                    if name:
                        slots.append((match,pos))

            def swap_into(target_pos, selected):
                current=target["players"][target_pos]
                if current==selected:
                    return
                source=next(((m,pos) for m,pos in slots if m["players"][pos]==selected),None)
                if source is None:
                    raise ValueError("Odabrani igrač nije pronađen u bracketu")
                source_match,source_pos=source
                source_match["players"][source_pos]=current
                target["players"][target_pos]=selected

            swap_into(0,a)
            swap_into(1,b)

            # Clear only derived results. The first-round slot structure and
            # every unaffected pairing remain exactly where they were.
            for match in self.state.get("matches",[]):
                match["winner"]=None; match["board"]=None
                match.pop("propagated",None)
                if int(match.get("round") or 0)==1:
                    present=[p for p in match.get("players",[]) if p]
                    match["status"]="ready" if len(present)>=int(match.get("expected") or 2) else "waiting"
                else:
                    match["players"]=[None,None]
                    match["status"]="waiting"
            self.state["winner"], self.state["status"] = None, "active"
            self._advance_byes(); self.save(); return self.snapshot()


class Host:
    def __init__(self, tournament, port=PORT):
        self.tournament,self.port=tournament,port; self.server=None
        self.discovery_socket=None; self._discovery_stop=threading.Event()
    def start(self):
        tournament=self.tournament
        class Handler(BaseHTTPRequestHandler):
            def log_message(self,*_): pass
            def reply(self,code,data):
                raw=json.dumps(data,ensure_ascii=False).encode(); self.send_response(code)
                self.send_header("Content-Type","application/json"); self.send_header("Content-Length",str(len(raw)))
                self.end_headers(); self.wfile.write(raw)
            def do_GET(self): self.reply(200,tournament.snapshot()) if self.path=="/state" else self.reply(404,{})
            def do_POST(self):
                try:
                    data=json.loads(self.rfile.read(int(self.headers.get("Content-Length",0))) or b"{}")
                    action=self.path.strip("/")
                    result=getattr(tournament,action)(**data) if action in ("claim","heartbeat","cancel","finish","assign","shuffle","undo","rename") else None
                    self.reply(200,result or {})
                except Exception as e: self.reply(409,{"error":str(e)})
        self.server=ThreadingHTTPServer(("0.0.0.0",self.port),Handler)
        threading.Thread(target=self.server.serve_forever,daemon=True,name="tournament-host").start()
        self._start_discovery()
    def _start_discovery(self):
        tournament=self.tournament; stop=self._discovery_stop
        def run():
            sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            sock.bind(("",DISCOVERY_PORT)); sock.settimeout(.5)
            self.discovery_socket=sock
            while not stop.is_set():
                try: data,addr=sock.recvfrom(512)
                except socket.timeout: continue
                except OSError: break
                if data==b"SMARTDARTS_TOURNAMENT_DISCOVER":
                    state=tournament.snapshot()
                    payload=json.dumps({"id":state.get("id"),"name":state.get("name","SMARTDARTS TURNIR")}).encode()
                    sock.sendto(payload,addr)
            sock.close()
        threading.Thread(target=run,daemon=True,name="tournament-discovery").start()
    def stop(self):
        self._discovery_stop.set()
        if self.discovery_socket is not None: self.discovery_socket.close(); self.discovery_socket=None
        if self.server is not None:
            self.server.shutdown(); self.server.server_close(); self.server=None


class Client:
    def __init__(self, host): self.base=f"http://{host}:{PORT}"
    def call(self,action="state",**data):
        try:
            if action=="state": return json.loads(urlopen(self.base+"/state",timeout=2).read())
            req=Request(self.base+"/"+action,json.dumps(data).encode(),{"Content-Type":"application/json"},method="POST")
            return json.loads(urlopen(req,timeout=3).read())
        except Exception as exc:
            # HTTP 409 responses contain the useful host-side reason. Surface
            # that text on the remote board instead of a generic HTTP error.
            try:
                payload=json.loads(exc.read().decode())
                if payload.get("error"):
                    raise ValueError(payload["error"])
            except AttributeError:
                pass
            except (UnicodeDecodeError,json.JSONDecodeError):
                pass
            raise
    @staticmethod
    def discover(timeout=2.0):
        sock=socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET,socket.SO_BROADCAST,1); sock.settimeout(.2)
        sock.bind(("",0)); sock.sendto(b"SMARTDARTS_TOURNAMENT_DISCOVER",("255.255.255.255",DISCOVERY_PORT))
        found={}; until=time.monotonic()+float(timeout)
        while time.monotonic()<until:
            try: raw,addr=sock.recvfrom(2048)
            except socket.timeout: continue
            try: info=json.loads(raw.decode()); found[addr[0]]={**info,"host":addr[0]}
            except Exception: pass
        sock.close(); return list(found.values())
