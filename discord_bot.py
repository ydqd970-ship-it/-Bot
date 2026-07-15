"""
株価分析 Discord Bot
使い方：
  !分析 7203.T       → 日本株（ティッカー）
  !分析 AAPL         → 米国株（ティッカー）
  !分析 トヨタ        → 銘柄名で検索
  !分析 7203.T NVDA  → 複数銘柄同時分析（最大3件）
  !ヘルプ            → 使い方表示
"""

import os, json, math, time, asyncio, warnings
import discord
from discord.ext import commands
import yfinance as yf
import numpy as np
import pandas as pd
import requests
import pytz
from datetime import datetime
warnings.filterwarnings("ignore")

# ===== 設定 =====
DISCORD_BOT_TOKEN  = os.environ.get("DISCORD_BOT_TOKEN", "")
DEEPSEEK_API_KEY   = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_API_URL   = "https://api.deepseek.com/chat/completions"

# 銘柄名→ティッカー検索辞書（日本株主要銘柄）
NAME_TO_TICKER = {
    "トヨタ": "7203.T", "トヨタ自動車": "7203.T",
    "ソニー": "6758.T", "ソニーグループ": "6758.T",
    "ソフトバンク": "9984.T", "ソフトバンクグループ": "9984.T",
    "キーエンス": "6861.T",
    "三菱UFJ": "8306.T", "みずほ": "8411.T",
    "信越化学": "4063.T",
    "ダイキン": "6367.T",
    "NTT": "9432.T", "日本電信電話": "9432.T",
    "ファナック": "6954.T",
    "任天堂": "7974.T",
    "東京エレクトロン": "8035.T",
    "中外製薬": "4519.T",
    "ファーストリテイリング": "9983.T", "ユニクロ": "9983.T",
    "リクルート": "6098.T",
    "ホンダ": "7267.T", "本田技研": "7267.T",
    "三井住友": "8316.T",
    "日立": "6501.T",
    "富士通": "6702.T",
    "TDK": "6762.T",
    "日本製鉄": "5401.T",
    "キャノン": "7751.T", "キヤノン": "7751.T",
    "三菱電機": "6503.T",
    "オリエンタルランド": "4661.T",
    "三菱商事": "8058.T",
    "セブンアイ": "3382.T", "セブン&アイ": "3382.T",
    "KDDI": "9433.T",
    "オリックス": "8591.T",
    "東京海上": "8766.T",
}

# セクター情報
SECTOR_MAP = {
    "8035.T":"半導体", "6762.T":"電子部品", "6971.T":"電子部品",
    "7203.T":"自動車", "7267.T":"自動車",
    "6758.T":"電機", "6861.T":"精密機器", "6954.T":"精密機器",
    "9984.T":"IT", "9432.T":"通信", "9433.T":"通信",
    "8306.T":"金融", "8316.T":"金融", "8411.T":"金融",
    "4063.T":"化学", "4519.T":"医薬品", "4502.T":"医薬品",
    "9983.T":"小売", "3382.T":"小売",
    "NVDA":"半導体", "AMD":"半導体", "INTC":"半導体",
    "AAPL":"IT", "MSFT":"IT", "GOOGL":"IT", "META":"IT",
    "AMZN":"小売・IT", "TSLA":"自動車",
    "JPM":"金融", "BAC":"金融", "GS":"金融",
    "JNJ":"医薬品", "UNH":"ヘルスケア",
}

# ===== ユーティリティ（stock_checker.pyと共通） =====

def _clean(values) -> list:
    if hasattr(values, 'squeeze'): values = values.squeeze()
    if hasattr(values, 'values'):  values = values.values
    result = []
    for v in values:
        try:
            f = float(v)
            if math.isfinite(f): result.append(f)
        except (TypeError, ValueError): pass
    return result

def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df = df.droplevel(0, axis=1) if df.columns.nlevels > 1 else df
    df.columns = [str(c).strip().capitalize() for c in df.columns]
    return df

def _fmt(value, fmt=".1f", fallback="--") -> str:
    try:
        f = float(value)
        if not math.isfinite(f): return fallback
        return format(f, fmt)
    except (TypeError, ValueError): return fallback

def _to_py(obj):
    if isinstance(obj, dict):   return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, list):   return [_to_py(v) for v in obj]
    if isinstance(obj, (np.integer,)):              return int(obj)
    if isinstance(obj, (np.floating, np.float32, np.float64)): return float(obj)
    if isinstance(obj, np.ndarray):                return obj.tolist()
    return obj

# ===== テクニカル指標 =====

def calc_rsi(closes, period=14):
    if len(closes) < period+1: return float("nan")
    gains, losses = [], []
    for i in range(len(closes)-period, len(closes)):
        diff = closes[i]-closes[i-1]
        (gains if diff>0 else losses).append(abs(diff))
    avg_gain = sum(gains)/period if gains else 0
    avg_loss = sum(losses)/period if losses else 0
    if avg_loss==0: return 100.0
    return round(100-100/(1+avg_gain/avg_loss), 2)

def calc_macd_hist(closes):
    if len(closes)<35: return float("nan")
    def ema(data,n):
        k=2/(n+1); r=[data[0]]
        for v in data[1:]: r.append(v*k+r[-1]*(1-k))
        return r
    ema12=ema(closes,12); ema26=ema(closes,26)
    macd=[a-b for a,b in zip(ema12,ema26)]
    return round(macd[-1]-ema(macd,9)[-1], 4)

def calc_bb(closes, period=20):
    if len(closes)<period: return float("nan"), float("nan")
    recent=closes[-period:]; mid=sum(recent)/period
    std=math.sqrt(sum((x-mid)**2 for x in recent)/period)
    upper,lower=mid+2*std,mid-2*std
    pos=(closes[-1]-lower)/(upper-lower)*100 if (upper-lower)!=0 else 50
    width=(upper-lower)/mid*100 if mid!=0 else 0
    return round(pos,2), round(width,2)

def calc_atr(highs,lows,closes,period=14):
    if len(closes)<period+1: return float("nan")
    trs=[max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1]))
         for i in range(len(closes)-period,len(closes))]
    return round(sum(trs)/period, 4)

def calc_adx(highs,lows,closes,period=14):
    if len(closes)<period*2+1: return float("nan")
    plus_dms,minus_dms,trs=[],[],[]
    for i in range(1,len(closes)):
        up=highs[i]-highs[i-1]; down=lows[i-1]-lows[i]
        plus_dms.append(up if up>down and up>0 else 0)
        minus_dms.append(down if down>up and down>0 else 0)
        trs.append(max(highs[i]-lows[i],abs(highs[i]-closes[i-1]),abs(lows[i]-closes[i-1])))
    def smooth(data,n):
        s=sum(data[:n]); r=[s]
        for v in data[n:]: r.append(r[-1]-r[-1]/n+v)
        return r
    sm_tr=smooth(trs,period); sm_p=smooth(plus_dms,period); sm_m=smooth(minus_dms,period)
    dx_list=[]
    for i in range(len(sm_tr)):
        if sm_tr[i]==0: continue
        pdi=100*sm_p[i]/sm_tr[i]; mdi=100*sm_m[i]/sm_tr[i]
        dx_list.append(100*abs(pdi-mdi)/(pdi+mdi) if (pdi+mdi)!=0 else 0)
    if len(dx_list)<period: return float("nan")
    return round(sum(dx_list[-period:])/period, 2)

def calc_ma_dev(closes,period):
    if len(closes)<period: return float("nan")
    ma=sum(closes[-period:])/period
    return round((closes[-1]-ma)/ma*100, 2) if ma!=0 else float("nan")

def calc_mom(closes,period):
    if len(closes)<=period: return float("nan")
    past=closes[-(period+1)]
    return round((closes[-1]-past)/past*100, 2) if past!=0 else float("nan")

def get_trend_label(p_up, p_down, p_range):
    if p_range is None: p_range=0.0
    if p_up is None:    p_up=0.0
    if p_down is None:  p_down=0.0
    if p_range>p_up and p_range>p_down: return "⚪", "レンジ圏"
    if p_up>=p_down:
        if p_up>=60:    return "🟢", "上昇優勢"
        elif p_up>=50:  return "🟢", "やや上昇優勢"
        else:           return "🟡", "弱い上昇優勢"
    else:
        if p_down>=60:  return "🔴", "下落優勢"
        elif p_down>=50:return "🔴", "やや下落優勢"
        else:           return "🟠", "弱い下落優勢"

# ===== データ取得・分析 =====

def resolve_ticker(query: str) -> str:
    """銘柄名またはティッカーを正規化して返す"""
    q = query.strip().upper()
    # 日本語名で検索
    for name, ticker in NAME_TO_TICKER.items():
        if query.strip() in name or name in query.strip():
            return ticker
    # そのままティッカーとして使用
    return q

def fetch_stock_data(ticker: str) -> dict | None:
    """株価データとテクニカル指標を取得"""
    try:
        raw = yf.download(ticker, period="1y",
                          auto_adjust=True, progress=False, threads=False)
        if raw.empty or len(raw) < 60:
            return None
        df = _normalize_df(raw).dropna()
        if len(df) < 60:
            return None

        closes  = _clean(df["Close"])
        highs   = _clean(df["High"])
        lows    = _clean(df["Low"])
        volumes = _clean(df["Volume"])

        current = closes[-1]
        prev    = closes[-2]
        change_pct = (current-prev)/prev*100 if prev!=0 else 0

        avg5 = sum(volumes[-5:])/5 if len(volumes)>=5 else 1
        vol_vs_avg = volumes[-1]/avg5*100 if avg5!=0 else 0
        vol_chg = ((volumes[-1]-volumes[-2])/volumes[-2]*100
                   if len(volumes)>=2 and volumes[-2]!=0 else 0)

        atr = calc_atr(highs, lows, closes)
        atr_pct = atr/current*100 if (math.isfinite(atr) and current!=0) else 0
        bb_pos, bb_width = calc_bb(closes)

        # 52週高値・安値
        week52_high = max(highs[-252:]) if len(highs)>=252 else max(highs)
        week52_low  = min(lows[-252:])  if len(lows) >=252 else min(lows)

        return {
            "ticker":      ticker,
            "current":     round(current, 2),
            "change_pct":  round(change_pct, 2),
            "volume":      int(volumes[-1]),
            "vol_vs_avg":  round(vol_vs_avg, 1),
            "vol_chg":     round(vol_chg, 1),
            "rsi":         calc_rsi(closes),
            "macd_hist":   calc_macd_hist(closes),
            "bb_pos":      bb_pos,
            "bb_width":    bb_width,
            "adx":         calc_adx(highs, lows, closes),
            "atr_pct":     round(atr_pct, 2),
            "ma_dev5":     calc_ma_dev(closes, 5),
            "ma_dev25":    calc_ma_dev(closes, 25),
            "ma_dev75":    calc_ma_dev(closes, 75),
            "mom5":        calc_mom(closes, 5),
            "mom20":       calc_mom(closes, 20),
            "mom60":       calc_mom(closes, 60),
            "week52_high": round(week52_high, 2),
            "week52_low":  round(week52_low, 2),
            "week52_pos":  round((current-week52_low)/(week52_high-week52_low)*100, 1)
                           if (week52_high-week52_low)!=0 else 50,
            "_closes": closes[-80:],
            "_highs":  highs[-80:],
            "_lows":   lows[-80:],
        }
    except Exception as e:
        return None

def run_xgboost(data: dict) -> dict:
    """
    XGBoostで簡易予測（過去1年のデータで訓練）。
    毎朝レポートより訓練データが少ないため簡易版。
    """
    try:
        from xgboost import XGBClassifier
        closes = data["_closes"]
        highs  = data["_highs"]
        lows   = data["_lows"]

        if len(closes) < 60:
            return {}

        # 特徴量生成
        def make_feats(c, h, l):
            bb_pos, bb_width = calc_bb(c)
            atr = calc_atr(h, l, c)
            atr_pct = atr/c[-1]*100 if (math.isfinite(atr) and c[-1]!=0) else 0
            return [
                calc_rsi(c) or 50,
                calc_macd_hist(c) or 0,
                bb_pos or 50, bb_width or 0,
                atr_pct or 0,
                calc_adx(h, l, c) or 20,
                calc_ma_dev(c,5) or 0,
                calc_ma_dev(c,25) or 0,
                calc_ma_dev(c,75) or 0,
                calc_mom(c,5) or 0,
                calc_mom(c,20) or 0,
            ]

        rows, labels = [], []
        for i in range(30, len(closes)-20):
            c=closes[:i+1]; h=highs[:i+1]; l=lows[:i+1]
            feats = make_feats(c, h, l)
            if any(not math.isfinite(v) for v in feats): continue
            ret = (closes[i+20]-closes[i])/closes[i]*100
            label = 0 if ret>=3 else (1 if ret<=-3 else 2)
            rows.append([v if math.isfinite(v) else 0 for v in feats])
            labels.append(label)

        if len(rows) < 20:
            return {}

        from lightgbm import LGBMClassifier
        from catboost import CatBoostClassifier

        X, y = np.array(rows), np.array(labels)
        feats_now = make_feats(closes, highs, lows)
        feats_now = [v if math.isfinite(v) else 0 for v in feats_now]
        x = np.array([feats_now])

        # 3モデルアンサンブル
        probas = []
        for ModelClass, kwargs in [
            (XGBClassifier, {"n_estimators":100,"max_depth":3,"learning_rate":0.1,
                             "eval_metric":"mlogloss","random_state":42,"n_jobs":-1}),
            (LGBMClassifier, {"n_estimators":100,"max_depth":3,"learning_rate":0.1,
                              "random_state":42,"n_jobs":-1,"verbose":-1}),
            (CatBoostClassifier, {"iterations":100,"depth":3,"learning_rate":0.1,
                                  "random_seed":42,"verbose":0}),
        ]:
            try:
                model = ModelClass(**kwargs)
                model.fit(X, y)
                probas.append(model.predict_proba(x)[0])
            except Exception:
                pass

        if not probas:
            return {}

        proba = np.mean(probas, axis=0)
        p_up    = round(float(proba[0])*100, 1)
        p_down  = round(float(proba[1])*100, 1)
        p_range = round(float(proba[2])*100, 1)

        # 期待値計算
        rets = [(closes[i+20]-closes[i])/closes[i]*100 for i in range(30, len(closes)-20)]
        up_r   = [r for r in rets if r>=3]
        down_r = [r for r in rets if r<=-3]
        rng_r  = [r for r in rets if -3<r<3]
        avg_up   = sum(up_r)/len(up_r)     if up_r   else 0
        avg_down = sum(down_r)/len(down_r) if down_r else 0
        avg_rng  = sum(rng_r)/len(rng_r)   if rng_r  else 0
        ev = p_up/100*avg_up + p_down/100*avg_down + p_range/100*avg_rng
        risk   = abs(p_down/100*avg_down)
        reward = p_up/100*avg_up
        rr = round(reward/risk, 2) if risk>0 else float("nan")

        return {
            "prob_up":    p_up,
            "prob_down":  p_down,
            "prob_range": p_range,
            "expected_value": round(ev, 2),
            "avg_up":   round(avg_up, 2),
            "avg_down": round(avg_down, 2),
            "risk_reward": rr,
        }
    except Exception as e:
        return {}

def calc_trade_plan(data: dict) -> dict:
    """ATR基準の売買プランを計算"""
    try:
        closes = data["_closes"]; highs = data["_highs"]; lows = data["_lows"]
        current = data["current"]
        atr = calc_atr(highs, lows, closes)
        if not math.isfinite(atr) or atr<=0: return {}
        support    = sum(sorted(lows[-20:])[:3])/3
        resistance = sum(sorted(highs[-20:], reverse=True)[:3])/3
        entry      = max(support, current-0.3*atr)
        stop_loss  = min(support-0.2*atr, current-1.0*atr)
        target1    = max(resistance, current+1.5*atr)
        target2    = current+2.5*atr
        risk       = abs(current-stop_loss)
        reward1    = abs(target1-current)
        rr         = round(reward1/risk, 2) if risk>0 else float("nan")
        return {
            "entry":      round(entry,    2),
            "stop_loss":  round(stop_loss,2),
            "target1":    round(target1,  2),
            "target2":    round(target2,  2),
            "atr_pct":    round(atr/current*100, 2),
            "risk_pct":   round(risk/current*100,    2),
            "reward_pct": round(reward1/current*100, 2),
            "rr_plan":    rr,
        }
    except Exception:
        return {}

def fetch_fundamental(ticker: str) -> dict:
    try:
        info = yf.Ticker(ticker).info
        per  = info.get("trailingPE")
        pbr  = info.get("priceToBook")
        roe  = info.get("returnOnEquity")
        eps  = info.get("earningsGrowth")
        name = info.get("longName") or info.get("shortName") or ticker
        return {
            "name": name,
            "per":  round(float(per),1)  if per and math.isfinite(float(per))  else None,
            "pbr":  round(float(pbr),2)  if pbr and math.isfinite(float(pbr))  else None,
            "roe":  round(float(roe)*100,1) if roe and math.isfinite(float(roe)) else None,
            "eps_growth": round(float(eps)*100,1) if eps and math.isfinite(float(eps)) else None,
        }
    except Exception:
        return {}

def _call_deepseek(prompt: str, max_tokens=1500) -> dict:
    headers = {"Authorization":f"Bearer {DEEPSEEK_API_KEY}","Content-Type":"application/json"}
    payload = {"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],
               "temperature":0.3,"max_tokens":max_tokens}
    try:
        resp = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=60)
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"): content=content[4:]
        return json.loads(content.strip())
    except Exception as e:
        return {"error": str(e)}

def analyze_ticker(ticker: str) -> str:
    """1銘柄の分析レポートを生成して返す"""
    # 1. データ取得
    data = fetch_stock_data(ticker)
    if data is None:
        return f"❌ `{ticker}` のデータを取得できませんでした。ティッカーを確認してください。"

    # 2. ファンダメンタル取得
    fund = fetch_fundamental(ticker)
    display_name = fund.get("name") or ticker

    # 3. XGBoostアンサンブル予測
    pred = run_xgboost(data)

    # 4. 売買プラン
    trade = calc_trade_plan(data)

    # 5. トレンドラベル
    icon, trend_text = get_trend_label(
        pred.get("prob_up"), pred.get("prob_down"), pred.get("prob_range"))

    # 6. DeepSeekでコメント生成
    jst     = pytz.timezone("Asia/Tokyo")
    now_str = datetime.now(jst).strftime("%Y/%m/%d %H:%M")
    sector  = SECTOR_MAP.get(ticker, "その他")

    prompt = f"""あなたは株式投資アドバイザーです。以下の銘柄を分析してください。

## 銘柄：{display_name}（{ticker}）
## テクニカル指標
{json.dumps(_to_py({**data, "_closes":None,"_highs":None,"_lows":None}),
            ensure_ascii=False, indent=2)}

## XGBoostアンサンブル予測（20日後）
{json.dumps(_to_py(pred), ensure_ascii=False, indent=2)}

## ファンダメンタル
{json.dumps(_to_py(fund), ensure_ascii=False, indent=2)}

## 依頼
テクニカル指標とXGBoost予測を総合して短い分析コメントを生成してください。

## 出力形式（JSONのみ）
{{
  "summary": "総合分析コメント（日本語・3〜4文）",
  "strength": "現在の強み（1文）",
  "risk": "リスク・注意点（1文）",
  "macro_note": "セクター・市場環境への一言（1文）"
}}"""

    ai = _call_deepseek(prompt, max_tokens=800)

    # 7. メッセージ組み立て
    change_arrow = "▲" if data["change_pct"]>=0 else "▼"
    price_str    = f"{data['current']:,.2f}"

    lines = [
        f"📊 **{display_name}（{ticker}）** 分析レポート　*{now_str}*",
        f"> セクター：{sector}　株価：{price_str}　{change_arrow}{abs(data['change_pct']):.2f}%",
        "",
    ]

    # AI総評
    if "error" not in ai:
        lines.append(f"💬 {ai.get('summary','')}")
        lines.append("")

    # テクニカル
    lines.append("📈 **テクニカル指標**")
    rsi = data.get("rsi"); adx = data.get("adx")
    rsi_note = "（買われすぎ）" if rsi and rsi>70 else "（売られすぎ）" if rsi and rsi<30 else ""
    adx_note = "（トレンド強）" if adx and adx>25 else "（レンジ）" if adx and adx<20 else ""
    lines.append(
        f"> RSI:{_fmt(rsi,'.1f')}{rsi_note}　ADX:{_fmt(adx,'.1f')}{adx_note}　"
        f"BB位置:{_fmt(data.get('bb_pos'),'.0f')}%"
    )
    lines.append(
        f"> 25日MA乖離:{_fmt(data.get('ma_dev25'),'+.1f')}%　"
        f"20日モメンタム:{_fmt(data.get('mom20'),'+.1f')}%　"
        f"出来高5日平均比:{_fmt(data.get('vol_vs_avg'),'.0f')}%"
    )
    lines.append(
        f"> 52週レンジ内位置:{_fmt(data.get('week52_pos'),'.0f')}%　"
        f"（高値:{data['week52_high']:,.2f} / 安値:{data['week52_low']:,.2f}）"
    )

    # XGBoost予測
    if pred:
        p_up   = pred.get("prob_up",0)
        p_down = pred.get("prob_down",0)
        p_rng  = pred.get("prob_range",0)
        ev     = pred.get("expected_value",0)
        rr     = pred.get("risk_reward")
        avg_up = pred.get("avg_up",0)
        avg_dn = pred.get("avg_down",0)

        ev_str  = f"+{ev:.2f}%" if ev>=0 else f"{ev:.2f}%"
        rr_str  = f"{rr:.2f}" if rr and math.isfinite(rr) else "--"

        lines.append("")
        lines.append("🤖 **XGBoostアンサンブル予測（20日後）**")
        lines.append(f"> {icon} トレンド見通し：**{trend_text}**")
        lines.append(
            f"> 上昇:{_fmt(p_up,'.1f')}%　下落:{_fmt(p_down,'.1f')}%　レンジ:{_fmt(p_rng,'.1f')}%"
        )
        lines.append(
            f"> 期待値：**{ev_str}**　（上昇時平均+{avg_up:.1f}% / 下落時平均{avg_dn:.1f}%）　RR:{rr_str}"
        )

    # 売買プラン
    if trade:
        lines.append("")
        lines.append(f"📋 **AI売買プラン**（ATR:{_fmt(trade.get('atr_pct'),'.2f')}% 基準・個人利用参考値）")
        lines.append(
            f"> 🎯 推奨エントリー：**{trade['entry']:,.2f}**　損切り：{trade['stop_loss']:,.2f}"
        )
        lines.append(
            f"> 📈 第一目標：{trade['target1']:,.2f}　第二目標：{trade['target2']:,.2f}"
        )
        lines.append(f"> リスク:{_fmt(trade.get('risk_pct'),'.1f')}%　リワード:{_fmt(trade.get('reward_pct'),'.1f')}%　プランRR:{_fmt(trade.get('rr_plan'),'.2f')}")

    # ファンダメンタル
    per=fund.get("per"); pbr=fund.get("pbr"); roe=fund.get("roe"); eps=fund.get("eps_growth")
    if any(v is not None for v in [per, pbr, roe]):
        def rate(v, low, high):
            if v is None: return "🔘--"
            if v<=low:    return f"🟢割安"
            elif v<=high: return f"🟡適正"
            else:         return f"🔴割高"
        def rate_roe(v):
            if v is None: return "🔘--"
            return "🟢高ROE" if v>=15 else ("🟡普通" if v>=8 else "🔴低ROE")

        lines.append("")
        lines.append("📊 **ファンダメンタル**（参考・長期指標）")
        per_str = f"PER:{per:.1f}倍（{rate(per,12,25)}）" if per else ""
        pbr_str = f"PBR:{pbr:.2f}倍（{rate(pbr,1.0,3.0)}）" if pbr else ""
        roe_str = f"ROE:{roe:.1f}%（{rate_roe(roe)}）" if roe else ""
        eps_str = f"EPS成長:+{eps:.1f}%" if eps and eps>=0 else (f"EPS成長:{eps:.1f}%" if eps else "")
        lines.append("> " + "　".join(filter(None, [per_str, pbr_str, roe_str, eps_str])))

    # AIリスク・強み
    if "error" not in ai:
        lines.append("")
        if ai.get("strength"): lines.append(f"✅ {ai['strength']}")
        if ai.get("risk"):     lines.append(f"⚠️ {ai['risk']}")
        if ai.get("macro_note"): lines.append(f"🌐 {ai['macro_note']}")

    lines.append("")
    lines.append("*※XGBoostは過去データの統計的傾向に基づく参考値です。投資は自己責任でお願いします。*")

    return "\n".join(lines)

# ===== Discord Bot =====

intents = discord.Intents.default()
intents.message_content = True
bot = commands.Bot(command_prefix="!", intents=intents)

@bot.event
async def on_ready():
    print(f"[INFO] Bot起動完了: {bot.user}")

@bot.command(name="分析")
async def analyze(ctx, *args):
    if not args:
        await ctx.send("❓ 使い方：`!分析 7203.T` または `!分析 AAPL` または `!分析 トヨタ`")
        return

    # 最大3銘柄まで
    queries = list(args[:3])
    await ctx.send(f"🔍 {len(queries)}銘柄を分析中です。しばらくお待ちください...")

    for query in queries:
        ticker = resolve_ticker(query)
        async with ctx.typing():
            # 分析は時間がかかるのでスレッドで実行
            loop = asyncio.get_event_loop()
            report = await loop.run_in_executor(None, analyze_ticker, ticker)

        # 2000文字制限で分割送信
        chunks = [report[i:i+1900] for i in range(0, len(report), 1900)]
        for chunk in chunks:
            await ctx.send(chunk)

        if len(queries) > 1:
            await asyncio.sleep(1)

@bot.command(name="ヘルプ")
async def help_cmd(ctx):
    msg = """📖 **株価分析Bot 使い方**

**コマンド：**
`!分析 [銘柄]` → 銘柄の詳細分析レポートを表示

**使用例：**
> `!分析 7203.T` → トヨタ自動車
> `!分析 AAPL` → Apple
> `!分析 トヨタ` → 銘柄名でも検索可能
> `!分析 TDK NVDA` → 複数銘柄（最大3件）

**表示内容：**
• テクニカル指標（RSI・ADX・BB・MA乖離等）
• XGBoostアンサンブル予測（20日後・3モデル）
• 期待値・リスクリワード比
• AI売買プラン（エントリー・損切り・目標）
• ファンダメンタル（PER・PBR・ROE）
• AIコメント（DeepSeek）

⚠️ *投資は自己責任でお願いします*"""
    await ctx.send(msg)

# Render.comのヘルスチェック用（スリープ防止）
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler

class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"OK")
    def log_message(self, format, *args): pass  # ログ抑制

def run_health_server():
    server = HTTPServer(("0.0.0.0", 8080), HealthHandler)
    server.serve_forever()

if __name__ == "__main__":
    if not DISCORD_BOT_TOKEN:
        print("[ERROR] DISCORD_BOT_TOKEN が設定されていません")
        exit(1)
    if not DEEPSEEK_API_KEY:
        print("[ERROR] DEEPSEEK_API_KEY が設定されていません")
        exit(1)

    # ヘルスチェックサーバーを別スレッドで起動（Render.com用）
    threading.Thread(target=run_health_server, daemon=True).start()
    print("[INFO] ヘルスチェックサーバー起動 :8080")

    bot.run(DISCORD_BOT_TOKEN)
