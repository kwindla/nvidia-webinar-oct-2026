import { test } from "node:test"
import assert from "node:assert/strict"
import { createStore } from "zustand/vanilla"
import { createSessionState } from "../src/stores/session-state.ts"

const catalog = (ids = ["cloud", "local"], defaultModel = "cloud") =>
  new Response(
    JSON.stringify({
      models: ids.map((id) => ({ id, name: id })),
      default: defaultModel,
    })
  )
const tick = () => new Promise((resolve) => setImmediate(resolve))
function deferred<T>() {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((r) => {
    resolve = r
  })
  return { promise, resolve }
}
function fixture(
  fetchCatalog: typeof fetch = async () => catalog(),
  timeout = 5000
) {
  const starts: unknown[] = []
  let state = "disconnected"
  let failStart = false
  let selection: unknown = {
    model: "cloud",
    available: ["cloud", "local"],
    names: { cloud: "Cloud", local: "Local" },
  }
  const client = {
    get state() {
      return state
    },
    needsInit: () => false,
    initDevices: async () => {},
    startBot: async (p: unknown) => {
      starts.push(p)
      if (failStart) throw Error("start failed")
      return {}
    },
    connect: async () => {
      state = "ready"
    },
    disconnect: async () => {
      state = "disconnecting"
      store.getState().sessionEnded()
      state = "disconnected"
    },
    sendClientRequest: async () => selection,
  }
  const store = createStore(
    createSessionState({
      getClient: () =>
        client as unknown as ReturnType<
          Parameters<typeof createSessionState>[0]["getClient"]
        >,
      onNewSession: () => {},
      fetchCatalog,
      catalogTimeoutMs: timeout,
    })
  )
  return {
    store,
    starts,
    setSelection: (v: unknown) => {
      selection = v
    },
    failStart: () => {
      failStart = true
    },
  }
}

test("initial catalog/default, explicit pick in nested start body, and visible named fallback", async () => {
  const f = fixture()
  await f.store.getState().loadModels()
  assert.equal(f.store.getState().model, "cloud")
  await f.store.getState().selectModel("local")
  await f.store.getState().connect()
  assert.deepEqual(f.starts[0], {
    endpoint: "/start",
    requestData: {
      transport: "webrtc",
      enableDefaultIceServers: true,
      body: { model: "local" },
    },
  })
  f.setSelection({
    model: "cloud",
    available: ["cloud"],
    names: { cloud: "Terra" },
    fallback: {
      requested: "Local Mini",
      actual: "Terra",
      reason: "The server timed out.",
    },
  })
  await f.store.getState().syncModel()
  assert.equal(f.store.getState().model, "cloud")
  assert.equal(
    f.store.getState().fallbackNotice,
    "Requested Local Mini; using Terra. The server timed out."
  )
  await f.store.getState().selectModel("cloud")
  assert.equal(f.store.getState().fallbackNotice, null)
  f.store.getState().dispose()
})

test("absent pick sends no model; malformed catalog can be retried", async () => {
  let fail = true
  const f = fixture(async () =>
    fail ? new Response("{}") : catalog(["cloud", "tuned"], "tuned")
  )
  await f.store.getState().loadModels()
  assert.ok(f.store.getState().catalogError)
  assert.equal(f.store.getState().catalogBusy, false)
  fail = false
  await f.store.getState().loadModels()
  assert.equal(f.store.getState().model, "tuned")
  assert.equal(f.store.getState().catalogError, null)
  await f.store.getState().connect()
  assert.deepEqual(
    (f.starts[0] as { requestData: { body: unknown } }).requestData.body,
    {}
  )
  f.store.getState().dispose()
})

test("bounded catalog timeout releases loading state, then retry succeeds", async () => {
  let stalled = true
  const f = fixture(async (_url, options) => {
    if (!stalled) return catalog()
    return await new Promise<Response>((_resolve, reject) =>
      options?.signal?.addEventListener(
        "abort",
        () => reject(Error("aborted")),
        { once: true }
      )
    )
  }, 10)
  await f.store.getState().loadModels()
  assert.equal(f.store.getState().catalogBusy, false)
  assert.ok(f.store.getState().catalogError)
  stalled = false
  await f.store.getState().loadModels()
  assert.equal(f.store.getState().catalogError, null)
  f.store.getState().dispose()
})

test("superseded catalog aborts and cannot overwrite a newer catalog or confirmed session", async () => {
  const old = deferred<Response>()
  let count = 0
  let signal: AbortSignal | null | undefined
  const f = fixture(async (_url, options) => {
    if (count++ === 0) {
      signal = options?.signal
      return old.promise
    }
    return catalog()
  })
  const pending = f.store.getState().loadModels()
  await f.store.getState().loadModels()
  assert.equal(signal?.aborted, true)
  await f.store.getState().connect()
  await f.store.getState().syncModel()
  old.resolve(catalog(["tuned"], "tuned"))
  await pending
  assert.equal(f.store.getState().model, "cloud")
  assert.equal(f.store.getState().modelBusy, false)
  f.store.getState().dispose()
})

test("failed start recovers catalog; disconnect refreshes and stale sync cannot overwrite reconnect", async () => {
  const f = fixture()
  await f.store.getState().loadModels()
  f.failStart()
  await f.store.getState().connect()
  await tick()
  assert.equal(f.store.getState().busy, false)
  assert.equal(f.store.getState().modelBusy, false)
  assert.equal(f.store.getState().catalogBusy, false)
  assert.match(f.store.getState().error!, /start failed/)
  f.store.getState().dispose()

  const g = fixture()
  await g.store.getState().loadModels()
  await g.store.getState().connect()
  const old = deferred<unknown>()
  g.setSelection(old.promise)
  const syncing = g.store.getState().syncModel()
  await g.store.getState().disconnect()
  await tick()
  assert.equal(g.store.getState().catalogBusy, false)
  g.setSelection({ model: "local", available: ["cloud", "local"] })
  await g.store.getState().connect()
  await g.store.getState().syncModel()
  old.resolve({ model: "cloud" })
  await syncing
  assert.equal(g.store.getState().model, "local")
  assert.equal(g.store.getState().modelBusy, false)
  g.store.getState().dispose()
})
